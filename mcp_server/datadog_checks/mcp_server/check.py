# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
"""
MCP server integration (Mode A: JSON-RPC protocol scrape).

The Agent acts as an MCP client over the streamable-HTTP transport: it performs
the `initialize` handshake, then queries the tool / resource / prompt catalogs,
measuring latency and counting catalog entries. It works against any
MCP-spec-compliant server with no server-side instrumentation.

This is the inverse of Datadog's Bits AI MCP Server (Datadog *as* an MCP
server): this integration monitors *your own* MCP servers. See README.

Namespace is ``mcp`` so metrics surface as ``mcp.server.*``.
"""

from __future__ import annotations

import time

from datadog_checks.base import AgentCheck, ConfigurationError

from .__about__ import __version__
from .client import JSONRPCError, MCPClient, MCPError

# The capabilities we enumerate each run: (capability key, JSON-RPC method,
# count metric). The capability key is also the field name in the result that
# holds the list of items.
CAPABILITIES = (
    ('tools', 'tools/list', 'server.tools.count'),
    ('resources', 'resources/list', 'server.resources.count'),
    ('prompts', 'prompts/list', 'server.prompts.count'),
)

TRANSPORT = 'http_sse'

# JSON-RPC "method not found": a server returns this for a capability it does
# not advertise. We treat it as "not supported" rather than an error.
METHOD_NOT_FOUND = -32601


class MCPServerCheck(AgentCheck):
    __NAMESPACE__ = 'mcp'

    SERVICE_CHECK_REACHABLE = 'server.reachable'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = self.instance.get('endpoint')
        if not self.endpoint:
            raise ConfigurationError('`endpoint` is required (e.g. http://localhost:8765/mcp)')
        self.server_name = self.instance.get('server_name', '')
        self._client_info = {'name': 'datadog-agent', 'version': __version__}
        # Catalog sizes from the previous run, for drift detection. In-memory,
        # so it resets on Agent restart — acceptable for v1.
        self._previous_counts = {}

    def check(self, _):
        base_tags = list(self.instance.get('tags', []))
        if self.server_name:
            base_tags.append(f'mcp_server:{self.server_name}')
        base_tags.append(f'transport:{TRANSPORT}')

        client = MCPClient(self.http, self.endpoint, self.log)

        # initialize — the one call whose failure is fatal for the run.
        try:
            init = self._timed_call(
                client.initialize,
                self._client_info,
                duration_metric='server.initialize.duration',
                tags=base_tags,
            )
        except JSONRPCError as e:
            self._emit_error(base_tags, 'initialize', e.code)
            self.service_check(
                self.SERVICE_CHECK_REACHABLE, AgentCheck.CRITICAL, tags=base_tags, message=str(e)[:200]
            )
            return
        except Exception as e:
            self.log.debug('initialize failed against %s', self.endpoint, exc_info=True)
            self.service_check(
                self.SERVICE_CHECK_REACHABLE, AgentCheck.CRITICAL, tags=base_tags, message=str(e)[:200]
            )
            return

        self.service_check(self.SERVICE_CHECK_REACHABLE, AgentCheck.OK, tags=base_tags)
        self._emit_protocol_version(init, base_tags)
        self._emit_catalogs(client, base_tags)

    def _emit_protocol_version(self, init, base_tags):
        """Emit a constant gauge of 1 carrying the negotiated protocol version
        and server identity as tags, so they are queryable in dashboards."""
        server_info = init.get('serverInfo', {}) or {}
        version_tags = base_tags + [
            f"protocol_version:{init.get('protocolVersion', 'unknown')}",
            f"server_name:{server_info.get('name', 'unknown')}",
            f"server_version:{server_info.get('version', 'unknown')}",
        ]
        self.gauge('server.protocol_version', 1, tags=version_tags)

    def _emit_catalogs(self, client, base_tags):
        """Query each capability's list endpoint, emit its count, and detect
        catalog drift across runs. A failure on one capability is logged and
        counted but does not stop the others."""
        for capability, method, count_metric in CAPABILITIES:
            try:
                result = self._timed_call(
                    client.call,
                    method,
                    {},
                    duration_metric='server.list.duration',
                    tags=base_tags,
                    duration_extra_tags=[f'capability:{capability}'],
                )
            except JSONRPCError as e:
                # A server that doesn't advertise a capability typically returns
                # "method not found": treat as "not supported" and skip silently
                # rather than counting it as an error.
                if e.code == METHOD_NOT_FOUND:
                    self.log.debug('%s not supported by server, skipping', method)
                    continue
                self.log.warning('MCP %s returned a JSON-RPC error: %s', method, e)
                self._emit_error(base_tags, method, e.code)
                continue
            except MCPError as e:
                self.log.warning('MCP %s failed: %s', method, e)
                self._emit_error(base_tags, method, type(e).__name__)
                continue
            except Exception as e:
                self.log.warning('MCP %s failed against %s: %s', method, self.endpoint, e)
                self._emit_error(base_tags, method, type(e).__name__)
                continue

            items = result.get(capability, []) or []
            count = len(items)
            self.gauge(count_metric, count, tags=base_tags)
            self._detect_drift(capability, count, base_tags)

    def _detect_drift(self, capability, count, base_tags):
        prev = self._previous_counts.get(capability)
        if prev is not None and prev != count:
            self.count('server.catalog_change.count', 1, tags=base_tags + [f'capability:{capability}'])
        self._previous_counts[capability] = count

    def _timed_call(self, fn, *args, duration_metric, tags, duration_extra_tags=None):
        """Invoke ``fn(*args)`` and, only on a successful round-trip, emit
        wall-clock latency (ms) under ``duration_metric``. On failure no latency
        is recorded — a connection error or JSON-RPC error is reported via the
        service check / error count instead, not as a misleading latency point."""
        t0 = time.monotonic()
        result = fn(*args)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self.gauge(duration_metric, elapsed_ms, tags=tags + (duration_extra_tags or []))
        return result

    def _emit_error(self, base_tags, method, error_code):
        self.count(
            'server.error.count',
            1,
            tags=base_tags + [f'jsonrpc_method:{method}', f'error_code:{error_code}'],
        )
