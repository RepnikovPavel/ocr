"""The tokens/s arithmetic behind the demo's live readout and every benchmark.

These numbers end up in reports and in decisions about what to optimize, so the
edge cases matter: a single token has no rate, an aborted run must not report a
finished one, and the decode rate must exclude the prefill it is not measuring.
"""

import time

from dots_mocr.utils.generation_stats import GenerationStats


def test_fresh_stats_report_nothing():
    stats = GenerationStats()
    assert stats.ttft_seconds is None
    assert stats.decode_tokens_per_second is None
    assert stats.total_tokens_per_second is None
    assert stats.to_dict()["generated_tokens"] == 0


def test_ttft_is_the_first_token_not_the_last():
    stats = GenerationStats().start(prompt_tokens=100)
    time.sleep(0.05)
    stats.record_token()
    first = stats.ttft_seconds
    time.sleep(0.05)
    stats.record_token()
    assert stats.ttft_seconds == first, "TTFT must not move once the first token lands"
    assert first >= 0.04


def test_decode_rate_excludes_the_prefill_token():
    """The first token carries the vision tower and prefill; folding it into the
    rate would understate generation speed on long pages."""
    stats = GenerationStats().start()
    stats.first_token_at = 10.0
    stats.last_token_at = 12.0
    stats.generated_tokens = 21
    # 20 intervals over 2 seconds, not 21 tokens over 2 seconds
    assert stats.decode_tokens_per_second == 10.0


def test_single_token_has_no_decode_rate():
    stats = GenerationStats().start()
    stats.record_token()
    stats.finish(generated_tokens=1)
    assert stats.decode_tokens_per_second is None, "one token spans no interval"
    assert stats.to_dict()["generated_tokens"] == 1


def test_finish_prefers_the_authoritative_token_count():
    """Stopping criteria miss the step that stops on EOS, so the decoded length
    wins over the hook's count."""
    stats = GenerationStats().start()
    for _ in range(5):
        stats.record_token()
    stats.finish(generated_tokens=7)
    assert stats.generated_tokens == 7


def test_wall_time_runs_while_generating_and_freezes_after():
    stats = GenerationStats().start()
    time.sleep(0.02)
    running = stats.wall_seconds
    assert running > 0
    time.sleep(0.02)
    assert stats.wall_seconds > running, "wall time must advance while in flight"
    stats.finish(generated_tokens=1)
    frozen = stats.wall_seconds
    time.sleep(0.02)
    assert stats.wall_seconds == frozen, "wall time must stop at finish()"


def test_aborted_is_carried_into_the_snapshot():
    stats = GenerationStats().start()
    stats.record_token()
    stats.finish(generated_tokens=1, aborted=True)
    assert stats.to_dict()["aborted"] is True


def test_snapshot_has_the_keys_the_ui_and_reports_read():
    stats = GenerationStats().start(prompt_tokens=42)
    for _ in range(3):
        stats.record_token()
    stats.finish(generated_tokens=3)
    snapshot = stats.to_dict()
    assert set(snapshot) == {
        "prompt_tokens", "generated_tokens", "ttft_seconds", "decode_seconds",
        "wall_seconds", "decode_tps", "total_tps", "aborted"}
    assert snapshot["prompt_tokens"] == 42
