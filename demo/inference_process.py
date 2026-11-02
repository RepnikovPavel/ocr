"""Disposable inference process; the API never imports the model runtime.

Only trusted parent/child processes share this private inherited pipe. No IPC
listener or user-supplied pickle is accepted. Exiting the process releases CUDA
contexts, CPU thread pools, model weights and allocator caches together.
"""

import importlib
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


class InferenceProcess:
    def __init__(self, *, startup_timeout=300, parser_class="dots_mocr.cli:DotsMOCRParser", **kwargs):
        self.abort_event = None
        self.generation_listener = None
        self.max_completion_tokens = kwargs.get("max_completion_tokens", 16384)
        checkpoint = kwargs.get("ckpt")
        self._checkpoint_files = (
            tuple(path for pattern in ("*.safetensors", "pytorch_model*.bin")
                  for path in Path(checkpoint).glob(pattern)) if checkpoint else ()
        )
        self.process = None
        self.connection, child = multiprocessing.Pipe(duplex=True)
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "demo.inference_process", str(child.fileno())],
                pass_fds=(child.fileno(),), start_new_session=True,
            )
            child.close()
            self.connection.send({"op": "load", "parser_class": parser_class, "kwargs": kwargs})
            loaded = self._receive(deadline=time.monotonic() + startup_timeout)
            self.device = loaded["device"]
            self.attn_implementation = loaded["attn_implementation"]
        except BaseException:
            child.close()
            self.close()
            raise

    @property
    def pid(self):
        return self.process.pid if self.process and self.process.poll() is None else None

    def _receive(self, deadline=None):
        abort_sent = False
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("inference process startup timed out")
            if self.abort_event is not None and self.abort_event.is_set() and not abort_sent:
                self.connection.send({"op": "cancel"})
                abort_sent = True
            if self.connection.poll(0.1):
                try:
                    message = self.connection.recv()
                except EOFError as error:
                    raise RuntimeError("inference process exited without a result") from error
                if message["type"] == "stats":
                    if self.generation_listener is not None:
                        from dots_mocr.utils.generation_stats import GenerationStats

                        stats = GenerationStats()
                        stats.__dict__.update(message["value"])
                        self.generation_listener(stats)
                    continue
                if message["type"] == "error":
                    raise RuntimeError(message["value"])
                return message["value"]
            if self.process.poll() is not None:
                raise RuntimeError(f"inference process exited with code {self.process.returncode}")

    def parse_page(self, **kwargs):
        try:
            self.connection.send({"op": "parse", "kwargs": kwargs,
                                  "max_completion_tokens": self.max_completion_tokens})
            return self._receive()
        except BaseException:
            self.close()
            raise

    def close(self):
        process = self.process
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            self.process = None
        self.connection.close()
        if hasattr(os, "posix_fadvise"):
            for path in self._checkpoint_files:
                try:
                    with path.open("rb") as weights:
                        os.posix_fadvise(weights.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError as error:
                    print(f"[inference] cannot release checkpoint cache: {error}", flush=True)


def serve(connection):
    commands = queue.Queue()
    replies = queue.Queue()
    abort = threading.Event()
    live = [None]

    def execute():
        parser = None
        while True:
            request = commands.get()
            try:
                if request["op"] == "load":
                    module, name = request["parser_class"].split(":")
                    parser = getattr(importlib.import_module(module), name)(**request["kwargs"])
                    parser.abort_event = abort
                    parser.generation_listener = lambda stats: live.__setitem__(0, stats)
                    result = {"device": parser.device,
                              "attn_implementation": getattr(parser, "attn_implementation", None)}
                else:
                    from demo.page_parser import parse_page

                    parser.max_completion_tokens = request["max_completion_tokens"]
                    result = parse_page(parser, **request["kwargs"])
                replies.put({"type": "result", "value": result})
            except BaseException as error:
                traceback.print_exc()
                replies.put({"type": "error", "value": f"{type(error).__name__}: {error}"})

    threading.Thread(target=execute, daemon=True, name="inference").start()
    busy = False
    while True:
        if connection.poll(0.1):
            try:
                request = connection.recv()
            except EOFError:
                return
            if request["op"] == "cancel":
                abort.set()
            else:
                if busy:
                    raise RuntimeError("concurrent inference commands are not supported")
                abort.clear()
                live[0] = None
                busy = True
                commands.put(request)
        stats = live[0]
        if busy and stats is not None:
            connection.send({"type": "stats", "value": dict(vars(stats))})
        try:
            reply = replies.get_nowait()
        except queue.Empty:
            continue
        connection.send(reply)
        busy = False
        if reply["type"] == "error":
            return


if __name__ == "__main__":
    from multiprocessing.connection import Connection

    try:
        serve(Connection(int(sys.argv[1])))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    finally:
        os._exit(0)
