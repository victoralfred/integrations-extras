# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
"""
A tiny JSON-RPC 2.0 client for talking to an MCP server over the
"streamable HTTP" transport (MCP 2024-11-05+).

It is intentionally thin: all HTTP concerns (auth, TLS, timeout, proxy,
headers) are delegated to the Agent's shared ``self.http`` wrapper, which the
check passes in. The client knows how to:

  * frame a JSON-RPC request/notification,
  * negotiate and carry the ``Mcp-Session-Id`` and ``MCP-Protocol-Version``
    headers across calls,
  * follow ``nextCursor`` pagination on list methods,
  * decode a response body whether the server replies with ``application/json``
    or an SSE ``text/event-stream`` event (FastMCP and the reference SDK both
    default to the latter), selecting the reply by JSON-RPC id, and
  * terminate the session when the run is done.

Timing and metric emission live in ``check.py``; this layer raises typed
exceptions and returns parsed results so the check can decide what to record.
"""

from __future__ import annotations

import json

# The protocol revision the Agent advertises during `initialize`. MCP servers
# negotiate down to a version they support and echo their choice back.
PROTOCOL_VERSION = '2024-11-05'

# Streamable HTTP servers may answer with either content type; we must accept
# both per the spec.
_ACCEPT = 'application/json, text/event-stream'

_SESSION_HEADER = 'Mcp-Session-Id'
# Required on every request after initialize since MCP spec 2025-06-18. Value is
# the protocol version negotiated during the handshake.
_PROTOCOL_VERSION_HEADER = 'MCP-Protocol-Version'

# Upper bound on pages followed for a single list method, so a server returning
# a never-ending (or repeating) cursor cannot loop the check forever.
MAX_PAGES = 100


class MCPError(Exception):
    """Base class for all client-side failures."""


class JSONRPCError(MCPError):
    """The server returned a well-formed JSON-RPC error object."""

    def __init__(self, code, message, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f'JSON-RPC error {code}: {message}')


class MalformedResponse(MCPError):
    """The response body was not parseable as a JSON-RPC message."""


class MCPClient:
    """Stateful JSON-RPC client. One instance per check run so the negotiated
    session id and protocol version are reused across ``initialize`` and the
    subsequent list calls, then discarded when ``terminate`` is called."""

    def __init__(self, http, endpoint, log):
        self.http = http
        self.endpoint = endpoint
        self.log = log
        self._session_id = None
        self._protocol_version = None
        self._request_id = 0

    def initialize(self, client_info):
        """Perform the MCP `initialize` handshake and return the server's
        ``result`` object. Captures the negotiated protocol version so it is
        attached to every subsequent request. The mandatory
        ``notifications/initialized`` follow-up is sent separately by
        ``confirm_initialized`` so it is not counted in initialize latency."""
        result = self.call(
            'initialize',
            {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {},
                'clientInfo': client_info,
            },
        )
        # Remember the version the server negotiated (fall back to what we
        # advertised) so post-initialize requests can carry the protocol header.
        self._protocol_version = result.get('protocolVersion') or PROTOCOL_VERSION
        return result

    def confirm_initialized(self):
        """Send the mandatory `notifications/initialized` confirmation. Per spec
        the client must confirm initialization before issuing other requests.
        It's a notification (no id, no response expected); non-fatal if a server
        does not require it."""
        try:
            self.notify('notifications/initialized')
        except MCPError as e:
            self.log.debug('notifications/initialized failed (continuing): %s', e)

    def list_all(self, method, item_key):
        """Call a paginated list method (tools/list, resources/list,
        prompts/list), following ``result.nextCursor`` across pages, and return
        ``(items, capped)`` where ``items`` is the concatenation of every page's
        ``item_key`` list and ``capped`` is True if pagination was stopped early
        by the page cap or a repeating cursor (counts may be truncated)."""
        items = []
        cursor = None
        seen_cursors = set()
        for _ in range(MAX_PAGES):
            params = {'cursor': cursor} if cursor else {}
            result = self.call(method, params)
            items.extend(result.get(item_key, []) or [])
            cursor = result.get('nextCursor')
            if not cursor:
                return items, False
            if cursor in seen_cursors:
                self.log.warning('MCP %s returned a repeating cursor; stopping pagination', method)
                return items, True
            seen_cursors.add(cursor)
        self.log.warning('MCP %s exceeded MAX_PAGES (%d); counts may be truncated', method, MAX_PAGES)
        return items, True

    def call(self, method, params=None):
        """Send a JSON-RPC request and return its ``result`` object.

        Raises:
            JSONRPCError: the server returned a JSON-RPC ``error``.
            MalformedResponse: the body could not be parsed.
            requests exceptions / HTTPError: connection or non-2xx status
                (propagated from the HTTP wrapper).
        """
        self._request_id += 1
        request_id = self._request_id
        body = {
            'jsonrpc': '2.0',
            'id': request_id,
            'method': method,
        }
        if params is not None:
            body['params'] = params

        response = self.http.post(self.endpoint, json=body, extra_headers=self._headers())
        response.raise_for_status()
        self._capture_session(response)

        data = self._decode(response, request_id)
        if not isinstance(data, dict):
            raise MalformedResponse(f'expected a JSON-RPC object, got {type(data).__name__}')
        if 'error' in data:
            err = data['error'] or {}
            raise JSONRPCError(err.get('code', 'unknown'), err.get('message', ''), err.get('data'))
        return data.get('result', {})

    def notify(self, method, params=None):
        """Send a JSON-RPC notification (no id, no result expected)."""
        body = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            body['params'] = params
        response = self.http.post(self.endpoint, json=body, extra_headers=self._headers())
        response.raise_for_status()
        self._capture_session(response)

    def terminate(self):
        """Best-effort session teardown so the monitored server does not
        accumulate one session per collection interval. MCP servers SHOULD honor
        an HTTP DELETE carrying the session header; some return 404/405 (client
        termination disallowed). Never inspects the body and never raises —
        teardown must not fail the check."""
        if not self._session_id:
            return
        try:
            self.http.delete(self.endpoint, extra_headers=self._headers())
        except Exception as e:
            self.log.debug('session termination failed (ignored): %s', e)

    def _headers(self):
        headers = {'Accept': _ACCEPT}
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        # Set only once initialize has negotiated a version, so the initialize
        # request itself carries no protocol header (per spec) and every request
        # after it does.
        if self._protocol_version:
            headers[_PROTOCOL_VERSION_HEADER] = self._protocol_version
        return headers

    def _capture_session(self, response):
        session_id = response.headers.get(_SESSION_HEADER)
        if session_id:
            self._session_id = session_id

    def _decode(self, response, request_id):
        """Decode the JSON-RPC reply from either a plain JSON body or an SSE
        ``text/event-stream`` body. For SSE, select the event whose JSON-RPC id
        matches the request we sent."""
        content_type = response.headers.get('Content-Type', '')
        if 'text/event-stream' in content_type:
            match = self._select_sse_message(response.text, request_id)
            if match is None:
                raise MalformedResponse(f'SSE response had no JSON-RPC message with id {request_id}')
            return match
        try:
            return response.json()
        except ValueError as e:
            raise MalformedResponse(f'response body was not valid JSON: {e}') from e

    @staticmethod
    def _select_sse_message(text, request_id):
        """Parse an SSE stream and return the first JSON object whose ``id``
        matches ``request_id``. Each event's ``data:`` lines are concatenated
        with '\\n' (per the SSE spec) before JSON parsing, so multi-line
        payloads decode correctly; events without a matching id (e.g. server
        notifications interleaved before the reply) are skipped. Comparison is
        lenient (string form) to tolerate servers that stringify the id."""
        target = str(request_id)
        data_lines = []

        def parse_event(buf):
            if not buf:
                return None
            try:
                obj = json.loads('\n'.join(buf))
            except ValueError:
                return None
            if isinstance(obj, dict) and 'id' in obj and str(obj['id']) == target:
                return obj
            return None

        for line in text.splitlines():
            if line.startswith('data:'):
                # Strip exactly one leading space after the colon (SSE spec),
                # preserving any further whitespace inside the payload.
                value = line[len('data:') :]
                if value.startswith(' '):
                    value = value[1:]
                data_lines.append(value)
            elif line == '':
                # Blank line terminates the current event.
                hit = parse_event(data_lines)
                if hit is not None:
                    return hit
                data_lines = []
            # Other SSE fields (event:, id:, retry:, comments) are ignored.
        # A trailing event with no final blank line.
        return parse_event(data_lines)
