# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
from datadog_checks.base import AgentCheck
from datadog_checks.dev.utils import get_metadata_metrics
from datadog_checks.mcp_server import MCPServerCheck
from datadog_checks.mcp_server import check as check_module

from .common import (
    BASE_TAGS,
    SERVER_PROTOCOL,
    SESSION_ID,
    FakeHttp,
    default_routes,
    error_response,
    malformed_response,
    page,
    sse_response,
)


def build_check(instance, fake_http):
    check = MCPServerCheck('mcp_server', {}, [instance])
    check._http = fake_http  # inject over the lazily-built RequestsWrapper
    return check


# --- happy path ------------------------------------------------------------


def test_happy_path_emits_all(aggregator, dd_run_check, instance):
    fake = FakeHttp(default_routes(tools=[{'n': 'a'}, {'n': 'b'}], resources=[{'r': 1}], prompts=[]))
    dd_run_check(build_check(instance, fake))

    aggregator.assert_service_check('mcp.server.reachable', status=AgentCheck.OK)
    aggregator.assert_metric('mcp.server.protocol_version', value=1)
    aggregator.assert_metric('mcp.server.tools.count', value=2)
    aggregator.assert_metric('mcp.server.resources.count', value=1)
    aggregator.assert_metric('mcp.server.prompts.count', value=0)
    aggregator.assert_metric('mcp.server.initialize.duration')
    aggregator.assert_metric('mcp.server.list.duration')
    aggregator.assert_all_metrics_covered()
    aggregator.assert_metrics_using_metadata(get_metadata_metrics())


# --- fix A: pagination -----------------------------------------------------


def test_pagination_sums_pages(aggregator, dd_run_check, instance):
    routes = default_routes()
    routes['tools/list'] = [
        page('tools', [{'n': 'a'}, {'n': 'b'}], next_cursor='c1'),
        page('tools', [{'n': 'c'}, {'n': 'd'}], next_cursor='c2'),
        page('tools', [{'n': 'e'}]),
    ]
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_metric('mcp.server.tools.count', value=5)
    tools_posts = fake.posts_for('tools/list')
    assert len(tools_posts) == 3
    # Pages 2 and 3 must carry the cursor returned by the previous page.
    assert tools_posts[0][1].get('params', {}).get('cursor') is None
    assert tools_posts[1][1]['params']['cursor'] == 'c1'
    assert tools_posts[2][1]['params']['cursor'] == 'c2'


def test_repeating_cursor_capped(aggregator, dd_run_check, instance, caplog):
    routes = default_routes()
    # Always returns the same cursor; the seen-cursor guard must stop it.
    routes['tools/list'] = page('tools', [{'n': 'a'}], next_cursor='loop')
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    # Two pages were fetched before the repeat was detected, then it stopped.
    assert len(fake.posts_for('tools/list')) == 2
    aggregator.assert_metric('mcp.server.tools.count', value=2)
    assert 'truncated' in caplog.text or 'repeating cursor' in caplog.text


# --- fix B: session teardown ----------------------------------------------


def test_session_delete_issued(aggregator, dd_run_check, instance):
    fake = FakeHttp(default_routes())
    dd_run_check(build_check(instance, fake))

    assert len(fake.deletes) == 1
    _, headers = fake.deletes[0]
    assert headers.get('Mcp-Session-Id') == SESSION_ID


def test_delete_failure_tolerated(aggregator, dd_run_check, instance):
    fake = FakeHttp(default_routes(tools=[{'n': 'a'}]))
    fake.delete_error = RuntimeError('HTTP 405')  # server disallows client termination
    dd_run_check(build_check(instance, fake))  # must not raise

    aggregator.assert_service_check('mcp.server.reachable', status=AgentCheck.OK)
    aggregator.assert_metric('mcp.server.tools.count', value=1)


# --- fix C: protocol version header ---------------------------------------


def test_protocol_header_on_post_init_not_initialize(aggregator, dd_run_check, instance):
    fake = FakeHttp(default_routes())
    dd_run_check(build_check(instance, fake))

    header = 'MCP-Protocol-Version'
    for _, body, headers in fake.posts:
        if body.get('method') == 'initialize':
            assert header not in headers
        else:
            assert headers.get(header) == SERVER_PROTOCOL


# --- fix D: robust SSE parsing --------------------------------------------


def test_sse_multiline_multievent_select_by_id(aggregator, dd_run_check, instance):
    routes = default_routes()
    # A notification event (no id) precedes the reply, whose data spans lines.
    routes['tools/list'] = sse_response(
        {'tools': [{'n': 'a'}, {'n': 'b'}, {'n': 'c'}]}, multiline=True, with_notification=True
    )
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_metric('mcp.server.tools.count', value=3)


# --- fix E: initialize-only timing ----------------------------------------


def test_initialize_timing_excludes_notification(aggregator, dd_run_check, instance, monkeypatch):
    # Stepping clock: (t0, t1) consumed per _timed_call. Only initialize and the
    # three list calls are timed; notifications/initialized is not.
    ticks = iter([0.0, 0.5, 10.0, 10.1, 20.0, 20.1, 30.0, 30.1])
    monkeypatch.setattr(check_module.time, 'monotonic', lambda: next(ticks))

    fake = FakeHttp(default_routes())
    dd_run_check(build_check(instance, fake))

    # 500ms == only the initialize round-trip, proving notify is outside the timer.
    aggregator.assert_metric('mcp.server.initialize.duration', value=500.0)
    # The confirmation was still sent, just untimed.
    assert fake.posts_for('notifications/initialized')


# --- backward-compatible error handling -----------------------------------


def test_method_not_found_skipped(aggregator, dd_run_check, instance):
    routes = default_routes(tools=[{'n': 'a'}])
    routes['resources/list'] = error_response(-32601, 'method not found')
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_metric('mcp.server.tools.count', value=1)
    aggregator.assert_metric('mcp.server.resources.count', count=0)
    aggregator.assert_metric('mcp.server.error.count', count=0)


def test_real_jsonrpc_error_counted(aggregator, dd_run_check, instance):
    routes = default_routes(tools=[{'n': 'a'}], resources=[{'r': 1}])
    routes['prompts/list'] = error_response(-32000, 'boom')
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_metric('mcp.server.tools.count', value=1)
    aggregator.assert_metric('mcp.server.resources.count', value=1)
    aggregator.assert_metric(
        'mcp.server.error.count',
        value=1,
        tags=BASE_TAGS + ['jsonrpc_method:prompts/list', 'error_code:-32000'],
    )


def test_malformed_body_counted(aggregator, dd_run_check, instance):
    routes = default_routes(resources=[{'r': 1}])
    routes['tools/list'] = malformed_response()
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_metric('mcp.server.resources.count', value=1)
    aggregator.assert_metric(
        'mcp.server.error.count',
        value=1,
        tags=BASE_TAGS + ['jsonrpc_method:tools/list', 'error_code:MalformedResponse'],
    )


def test_connection_error_critical(aggregator, dd_run_check, instance):
    routes = default_routes()
    routes['initialize'] = ConnectionError('connection refused')
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_service_check('mcp.server.reachable', status=AgentCheck.CRITICAL)
    aggregator.assert_metric('mcp.server.tools.count', count=0)
    aggregator.assert_metric('mcp.server.protocol_version', count=0)
    assert fake.deletes == []  # no session was established, so nothing to tear down


def test_initialize_jsonrpc_error_critical(aggregator, dd_run_check, instance):
    routes = default_routes()
    routes['initialize'] = error_response(-32602, 'invalid params')
    fake = FakeHttp(routes)
    dd_run_check(build_check(instance, fake))

    aggregator.assert_service_check('mcp.server.reachable', status=AgentCheck.CRITICAL)
    aggregator.assert_metric(
        'mcp.server.error.count',
        value=1,
        tags=BASE_TAGS + ['jsonrpc_method:initialize', 'error_code:-32602'],
    )
    aggregator.assert_metric('mcp.server.tools.count', count=0)
    assert fake.deletes == []


# --- drift detection across runs ------------------------------------------


def test_catalog_drift_across_runs(aggregator, dd_run_check, instance):
    tools5 = [{'n': i} for i in range(5)]
    tools6 = [{'n': i} for i in range(6)]
    check = build_check(instance, FakeHttp(default_routes(tools=tools5)))

    dd_run_check(check)  # first run establishes the baseline (no drift)
    check._http = FakeHttp(default_routes(tools=tools6))
    dd_run_check(check)  # second run: tools count changed 5 -> 6

    aggregator.assert_metric(
        'mcp.server.catalog_change.count',
        value=1,
        count=1,
        tags=BASE_TAGS + ['capability:tools'],
    )


def test_endpoint_required(instance):
    import pytest

    from datadog_checks.base import ConfigurationError

    bad = dict(instance)
    bad.pop('endpoint')
    with pytest.raises(ConfigurationError):
        MCPServerCheck('mcp_server', {}, [bad])
