# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
"""
Test doubles for the MCP HTTP transport.

``FakeHttp`` stands in for the Agent's ``self.http`` wrapper: it routes POSTs by
JSON-RPC method to canned responses (a single response, a callable that builds
one from the request body, a list consumed page-by-page, or an exception to
raise), and records every POST/DELETE so tests can assert on headers and bodies.
"""

import json

SERVER_PROTOCOL = '2025-06-18'
SESSION_ID = 'sess-abc'

INSTANCE = {
    'endpoint': 'http://localhost:8765/mcp',
    'server_name': 'test',
    'tags': ['foo:bar'],
}

# base_tags the check builds from INSTANCE, for convenience in assertions.
BASE_TAGS = ['foo:bar', 'mcp_server:test', 'transport:http_sse']


class FakeResponse:
    def __init__(self, status_code=200, headers=None, json_body=None, text=''):
        self.status_code = status_code
        self.headers = headers or {}
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError('no JSON body')
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')


class FakeHttp:
    """Routes POSTs by JSON-RPC method; records calls for assertions."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.posts = []  # list of (url, body, headers)
        self.deletes = []  # list of (url, headers)
        self.delete_response = FakeResponse(status_code=200)
        self.delete_error = None  # set to an exception to simulate DELETE failure

    def post(self, url, json=None, extra_headers=None):
        body = json
        self.posts.append((url, body, dict(extra_headers or {})))
        route = self.routes[body.get('method')]
        if isinstance(route, list):
            route = route.pop(0)
        if isinstance(route, BaseException):
            raise route
        return route(body) if callable(route) else route

    def delete(self, url, extra_headers=None):
        self.deletes.append((url, dict(extra_headers or {})))
        if self.delete_error is not None:
            raise self.delete_error
        return self.delete_response

    def posts_for(self, method):
        return [p for p in self.posts if p[1].get('method') == method]


# --- response builders -----------------------------------------------------


def json_response(result, *, status=200, session_id=None):
    """A callable returning an application/json JSON-RPC result echoing the id."""
    headers = {'Content-Type': 'application/json'}
    if session_id:
        headers['Mcp-Session-Id'] = session_id

    def make(body):
        return FakeResponse(
            status_code=status,
            headers=dict(headers),
            json_body={'jsonrpc': '2.0', 'id': body.get('id'), 'result': result},
        )

    return make


def error_response(code, message='boom'):
    def make(body):
        return FakeResponse(
            status_code=200,
            headers={'Content-Type': 'application/json'},
            json_body={'jsonrpc': '2.0', 'id': body.get('id'), 'error': {'code': code, 'message': message}},
        )

    return make


def malformed_response():
    def make(body):
        return FakeResponse(
            status_code=200, headers={'Content-Type': 'application/json'}, json_body=None, text='not json'
        )

    return make


def page(item_key, items, next_cursor=None):
    result = {item_key: items}
    if next_cursor:
        result['nextCursor'] = next_cursor
    return json_response(result)


def sse_response(result, *, multiline=False, with_notification=False):
    """Build a text/event-stream reply. Optionally prepend a notification event
    (no id, must be skipped) and split the reply's data across multiple lines."""
    headers = {'Content-Type': 'text/event-stream'}

    def make(body):
        events = []
        if with_notification:
            note = {'jsonrpc': '2.0', 'method': 'notifications/message', 'params': {'level': 'info'}}
            events.append([json.dumps(note)])
        reply = {'jsonrpc': '2.0', 'id': body.get('id'), 'result': result}
        reply_json = json.dumps(reply, indent=2) if multiline else json.dumps(reply)
        events.append(reply_json.split('\n'))

        lines = []
        for data_lines in events:
            lines.append('event: message')
            for dl in data_lines:
                lines.append(f'data: {dl}')
            lines.append('')  # blank line terminates the event
        return FakeResponse(status_code=200, headers=dict(headers), text='\n'.join(lines))

    return make


def initialize_ok():
    return json_response(
        {
            'protocolVersion': SERVER_PROTOCOL,
            'serverInfo': {'name': 'srv', 'version': '9'},
            'capabilities': {},
        },
        session_id=SESSION_ID,
    )


def default_routes(tools=None, resources=None, prompts=None):
    """Standard happy-path routes; individual methods can be overridden."""
    return {
        'initialize': initialize_ok(),
        'notifications/initialized': FakeResponse(status_code=202),
        'tools/list': json_response({'tools': tools or []}),
        'resources/list': json_response({'resources': resources or []}),
        'prompts/list': json_response({'prompts': prompts or []}),
    }
