# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
"""
A tiny JSON-RPC 2.0 client for talking to an MCP server over the
"streamable HTTP" transport (MCP 2024-11-05+).

It is intentionally thin: all HTTP concerns (auth, TLS, timeout, proxy,
headers) are delegated to the Agent's shared ``self.http`` wrapper, which the
check passes in. The client only knows how to:

  * frame a JSON-RPC request/notification,
  * negotiate and carry the ``Mcp-Session-Id`` header across calls, and
  * decode a response body whether the server replies with ``application/json``
    or an SSE ``text/event-stream`` event (FastMCP and the reference SDK both
    default to the latter).

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
    session id is reused across ``initialize`` and the subsequent list calls,
    then discarded."""

    def __init__(self, http, endpoint, log):
        self.http = http
        self.endpoint = endpoint
        self.log = log
        self._session_id = None
        self._request_id = 0

    def initialize(self, client_info):
        """Perform the MCP `initialize` handshake and the mandatory
        `notifications/initialized` follow-up. Returns the server's
        ``result`` object."""
        result = self.call(
            'initialize',
            {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {},
                'clientInfo': client_info,
            },
        )
        # Per spec the client must confirm initialization before issuing other
        # requests. It's a notification (no id, no response expected).
        try:
            self.notify('notifications/initialized')
        except MCPError as e:
            # Non-fatal: some servers don't require it. Log and continue.
            self.log.debug('notifications/initialized failed (continuing): %s', e)
        return result

    def call(self, method, params=None):
        """Send a JSON-RPC request and return its ``result`` object.

        Raises:
            JSONRPCError: the server returned a JSON-RPC ``error``.
            MalformedResponse: the body could not be parsed.
            requests exceptions / HTTPError: connection or non-2xx status
                (propagated from the HTTP wrapper).
        """
        self._request_id += 1
        body = {
            'jsonrpc': '2.0',
            'id': self._request_id,
            'method': method,
        }
        if params is not None:
            body['params'] = params

        response = self.http.post(self.endpoint, json=body, extra_headers=self._headers())
        response.raise_for_status()
        self._capture_session(response)

        data = self._decode(response)
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

    def _headers(self):
        headers = {'Accept': _ACCEPT}
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        return headers

    def _capture_session(self, response):
        session_id = response.headers.get(_SESSION_HEADER)
        if session_id:
            self._session_id = session_id

    def _decode(self, response):
        """Decode a JSON-RPC message from either a plain JSON body or an SSE
        ``text/event-stream`` body (taking the last ``data:`` payload)."""
        content_type = response.headers.get('Content-Type', '')
        if 'text/event-stream' in content_type:
            payload = self._extract_sse_data(response.text)
            if payload is None:
                raise MalformedResponse('SSE response contained no data frame')
            try:
                return json.loads(payload)
            except ValueError as e:
                raise MalformedResponse(f'invalid JSON in SSE data frame: {e}') from e
        try:
            return response.json()
        except ValueError as e:
            raise MalformedResponse(f'response body was not valid JSON: {e}') from e

    @staticmethod
    def _extract_sse_data(text):
        """Return the JSON payload of the last ``data:`` line in an SSE stream,
        or None if there is none. A single response carries one JSON-RPC reply,
        so the last data frame is the message of interest."""
        last = None
        for line in text.splitlines():
            if line.startswith('data:'):
                last = line[len('data:'):].strip()
        return last
