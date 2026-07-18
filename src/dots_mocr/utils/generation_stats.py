"""Per-generation token telemetry: tokens/s for the demo UI and the benchmarks.

transformers calls every StoppingCriteria once per decoding step, right after the
new token has been appended to input_ids. So the call count is the number of
generated tokens and the first call marks time-to-first-token. Piggy-backing on
that hook keeps the measurement free: no streamer thread, no second forward pass,
and it works identically for greedy and sampled decoding.

The stats object is mutated in place while generation runs, so a reader in
another thread (the demo server rendering /api/state) can hold a reference and
poll it. Only plain int/float fields are written, and a torn read costs nothing
worse than a momentarily stale tokens/s reading.
"""

import time


class GenerationStats:
    """Timing/counter sink for a single model.generate() call."""

    def __init__(self):
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.started_at = None       # perf_counter at generate() entry
        self.first_token_at = None   # perf_counter when token #1 landed
        self.last_token_at = None
        self.finished_at = None
        self.aborted = False

    # ------------------------------------------------------------ recording

    def start(self, prompt_tokens=0):
        self.prompt_tokens = int(prompt_tokens)
        self.started_at = time.perf_counter()
        return self

    def record_token(self):
        now = time.perf_counter()
        if self.first_token_at is None:
            self.first_token_at = now
        self.generated_tokens += 1
        self.last_token_at = now

    def finish(self, generated_tokens=None, aborted=False):
        """Close the measurement.

        `generated_tokens` overrides the hook count with the authoritative
        length of the decoded sequence: transformers may stop on an EOS that the
        criteria never observed, and stopping criteria are not called at all when
        max_new_tokens is reached in a single step.
        """
        self.finished_at = time.perf_counter()
        self.aborted = bool(aborted)
        if generated_tokens is not None:
            self.generated_tokens = int(generated_tokens)
        return self

    # ------------------------------------------------------------ derived

    @property
    def ttft_seconds(self):
        """Prefill latency: image encode + prompt attention + first decode step."""
        if self.started_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.started_at

    @property
    def wall_seconds(self):
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.perf_counter()
        return end - self.started_at

    @property
    def decode_seconds(self):
        """Time spent in the decode loop, excluding the first (prefill) token."""
        if self.first_token_at is None or self.last_token_at is None:
            return None
        return self.last_token_at - self.first_token_at

    @property
    def decode_tokens_per_second(self):
        """Steady-state decode rate — the number users mean by "t/s".

        Excludes TTFT: the first token carries the whole vision tower + prefill,
        so folding it in would understate the generation speed on long pages and
        overstate nothing.
        """
        elapsed = self.decode_seconds
        if not elapsed or self.generated_tokens < 2:
            return None
        return (self.generated_tokens - 1) / elapsed

    @property
    def total_tokens_per_second(self):
        """End-to-end rate including prefill — what a caller feels per request."""
        elapsed = self.wall_seconds
        if not elapsed or not self.generated_tokens:
            return None
        return self.generated_tokens / elapsed

    def to_dict(self, ndigits=2):
        def rounded(value):
            return None if value is None else round(value, ndigits)

        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_seconds": rounded(self.ttft_seconds),
            "decode_seconds": rounded(self.decode_seconds),
            "wall_seconds": rounded(self.wall_seconds),
            "decode_tps": rounded(self.decode_tokens_per_second),
            "total_tps": rounded(self.total_tokens_per_second),
            "aborted": self.aborted,
        }
