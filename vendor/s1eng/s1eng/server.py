"""Minimal asyncio HTTP/1.1 server for the s1-fast engine contract (unix socket or TCP).

GET  /health                -> 200 {"ok":true,"model":"s1-fast","manifest":<weights revision>}  (HEAD -> 405, as FastAPI)
POST /internal/v1/decide    -> header X-S1M-Internal-Token (hmac compare), body {"state", "questions": {qid: q}}
                               200 {"model":"s1-fast","answers":{qid: answer},"usage":{"input_tokens":n,"output_tokens":0}}
Error bodies and status codes follow the reference server (plumb_server.py on FastAPI/uvicorn).
"""
from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import os
import time

import numpy as np

from .prompt import options_of, prompt_text


def _dumps(obj) -> bytes:
    # FastAPI JSONResponse.render
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


OPTION_COUNT = contextvars.ContextVar('option_count', default=16)

STATS: dict = {}   # S1ENG_TRACE: per-request timing sums (tokenize, submit -> result, result -> sent), read by the trace
TRACE = os.environ.get("S1ENG_TRACE", "0") != "0"


def host_delay_us_of(v) -> int:
    """host_delay_us (env S1ENG_HOST_DELAY_US / ARMS key): int(v) microseconds >= 0 (floats truncate), else ValueError."""
    us = int(v)
    if us < 0:
        raise ValueError(f"host_delay_us must be >= 0 (got {v!r})")
    return us


def env_host_delay_ns(env=os.environ) -> int:
    return host_delay_us_of(env.get("S1ENG_HOST_DELAY_US", "0") or "0") * 1000


# backlog #4 stage 0 (kai/kx-host, timing probe only): every decision busy-waits this many ns just before its submit
# (0 = today's engine: one int compare per request). The spin holds the GIL only while host_delay_us < switch_us: once
# another thread has waited one switch interval, the interpreter hands it the GIL mid-spin. BatchLoop._apply_arm sets
# it (ARMS key host_delay_us); the default is S1ENG_HOST_DELAY_US.
HOST_DELAY_NS = env_host_delay_ns()


def _spin(ns: int) -> None:
    """Busy-wait `ns` ns on this thread (no sleep, no I/O). Holds the GIL only while ns < the switch interval."""
    clk = time.perf_counter_ns
    end = clk() + ns
    while clk() < end:
        pass

_REASON = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed",
           413: "Request Entity Too Large", 500: "Internal Server Error"}


class Engine:
    """Request handling: everything except the GPU work, which goes through `gpu.submit`."""

    def __init__(self, encoder, gpu, temperature: float, token: str, max_tokens: int, revision: str):
        self.enc = encoder
        self.gpu = gpu
        self.temperature = temperature
        self.token = token
        self.max_tokens = max_tokens
        self.health_body = _dumps({"ok": True, "model": "s1-fast", "manifest": revision})
        self.loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ decide
    async def decide(self, headers: dict, body: bytes, tm: dict | None = None) -> tuple[int, bytes, str]:
        if self.token and not hmac.compare_digest(headers.get("x-s1m-internal-token", ""), self.token):
            return 401, _dumps({"error": {"message": "unauthorized"}}), "application/json"
        try:
            req = json.loads(body)
            (qid, q), = req["questions"].items()
            if q.get("type") not in ("choice", "noul", "score"):
                raise ValueError("bad question type")
            q.setdefault("instructions", "")
        except Exception as exc:  # noqa: BLE001
            return 400, _dumps({"error": {"message": f"bad request: {type(exc).__name__}"}}), "application/json"
        try:
            options = options_of(q)
            t0 = time.perf_counter() if TRACE else 0.0
            ids = self.enc.encode(prompt_text(req["state"], q["instructions"], [t for _, t in options]))
            if len(ids) > self.max_tokens:
                raise ValueError(f"input exceeds {self.max_tokens} model tokens")
            t1 = time.perf_counter() if TRACE else 0.0
            option_token = OPTION_COUNT.set(len(options))
            try:
                logits = await self._run(ids, tm)
            finally:
                OPTION_COUNT.reset(option_token)
            if TRACE:
                t2 = time.perf_counter()
                STATS["n"] = STATS.get("n", 0) + 1
                STATS["tok"] = STATS.get("tok", 0.0) + (t1 - t0)
                STATS["gpu"] = STATS.get("gpu", 0.0) + (t2 - t1)
                if tm is not None:
                    tm["res"] = t2
            answer = self._answer(q["type"], options, logits)
        except ValueError as exc:
            return 400, _dumps({"error": {"message": str(exc)}}), "application/json"
        except Exception:  # noqa: BLE001  (the reference lets these escape: uvicorn answers 500)
            return 500, b"Internal Server Error", "text/plain; charset=utf-8"
        return 200, _dumps({"model": "s1-fast", "answers": {qid: answer},
                            "usage": {"input_tokens": len(ids), "output_tokens": 0}}), "application/json"

    def _run(self, ids: list[int], tm: dict | None = None) -> asyncio.Future:
        from .model import Job

        loop = self.loop
        fut = loop.create_future()

        def done(res):
            loop.call_soon_threadsafe(_resolve, fut, res)

        if tm is not None and tm.pop("arr", False):   # ARRIVE: this request is queued now, no longer arriving
            self.gpu.arriving -= 1
        if HOST_DELAY_NS > 0:   # backlog #4 stage 0: the host-delay probe (arm key host_delay_us)
            _spin(HOST_DELAY_NS)
        job = Job(np.asarray(ids, dtype=np.int64), done)
        job.option_count = OPTION_COUNT.get()
        self.gpu.submit(job)
        return fut

    def _answer(self, kind: str, options, logits16: np.ndarray) -> dict:
        # plumb_server._decide_one: float32 logits of the first k letters / T, numpy softmax
        logits = logits16[None, :]
        row = logits[0, : len(options)] / self.temperature
        exp = np.exp(row - row.max())
        p = exp / exp.sum()
        probs = {key: float(prob) for (key, _), prob in zip(options, p)}
        answer = {"type": kind, "probabilities": probs, "confidence": max(probs.values())}
        if kind == "noul":
            answer["noul"] = probs["true"]
        elif kind == "choice":
            answer["choice"] = max(probs, key=probs.get)
        else:
            answer["score"] = sum(int(key) * prob for key, prob in probs.items())
        return answer

    # ------------------------------------------------------------------ HTTP
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                except asyncio.LimitOverrunError:
                    await self._send(writer, 413, b"", "text/plain", close=True)
                    return
                lines = head[:-4].decode("latin-1").split("\r\n")
                try:
                    method, target, version = lines[0].split(" ", 2)
                except ValueError:
                    await self._send(writer, 400, b"Invalid HTTP request", "text/plain", close=True)
                    return
                headers = {}
                for line in lines[1:]:
                    k, sep, v = line.partition(":")
                    if sep:
                        headers[k.strip().lower()] = v.strip()
                if headers.get("transfer-encoding", "").lower() == "chunked":
                    body = await self._read_chunked(reader)
                else:
                    n = int(headers.get("content-length", "0") or 0)
                    body = await reader.readexactly(n) if n else b""
                close = (headers.get("connection", "").lower() == "close") or version == "HTTP/1.0"
                path = target.split("?", 1)[0]
                allow = None
                tm = {}
                if path == "/internal/v1/decide":
                    if method == "POST":
                        if getattr(self.gpu, "arrive", False):
                            # ARRIVE (rgh, u198 idea): a decision between HTTP arrival and the queue, so the GPU
                            # loop's batch close knows more is on the way (this thread is the counter's only writer)
                            self.gpu.arriving += 1
                            tm["arr"] = True
                        try:
                            if tm.get("arr") and getattr(self.gpu, "arrive_yield", False):
                                # RAMP (kai/kx-sched, S1ENG_ARRIVE_YIELD / arm key arrive_yield, default off): one
                                # yield after the count, so the other requests of a burst are counted before this
                                # one tokenizes and queues (inside the try: a cancel still clears the count)
                                await asyncio.sleep(0)
                            status, out, ctype = await self.decide(headers, body, tm)
                        except Exception:  # noqa: BLE001
                            status, out, ctype = 500, b"Internal Server Error", "text/plain; charset=utf-8"
                        finally:
                            if tm.pop("arr", False):   # ARRIVE: answered without reaching the queue
                                self.gpu.arriving -= 1
                    else:
                        status, out, ctype, allow = 405, b'{"detail":"Method Not Allowed"}', "application/json", "POST"
                elif path == "/health":
                    # FastAPI routes do not add HEAD to GET routes: the reference answers HEAD /health with 405
                    if method == "GET":
                        status, out, ctype = 200, self.health_body, "application/json"
                    else:
                        status, out, ctype, allow = 405, b'{"detail":"Method Not Allowed"}', "application/json", "GET"
                else:
                    status, out, ctype = 404, b'{"detail":"Not Found"}', "application/json"
                await self._send(writer, status, out, ctype, close=close, head_only=(method == "HEAD"), allow=allow)
                if TRACE and "res" in tm:
                    STATS["send"] = STATS.get("send", 0.0) + (time.perf_counter() - tm["res"])
                if close:
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    async def _read_chunked(reader) -> bytes:
        parts = []
        while True:
            size = int((await reader.readuntil(b"\r\n"))[:-2].split(b";")[0], 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                return b"".join(parts)
            parts.append(await reader.readexactly(size))
            await reader.readexactly(2)

    @staticmethod
    async def _send(writer, status: int, body: bytes, ctype: str, close: bool = False, head_only: bool = False,
                    allow: str | None = None):
        # header order as uvicorn/FastAPI send them (allow first), without date / server
        hdr = (f"HTTP/1.1 {status} {_REASON.get(status, 'OK')}\r\n" + (f"allow: {allow}\r\n" if allow else "")
               + f"content-length: {len(body)}\r\ncontent-type: {ctype}\r\n"
               + ("connection: close\r\n" if close else "") + "\r\n").encode("latin-1")
        writer.write(hdr if head_only else hdr + body)
        await writer.drain()


def _resolve(fut: asyncio.Future, res) -> None:
    if fut.done():
        return
    if isinstance(res, BaseException):
        fut.set_exception(res)
    else:
        fut.set_result(res)


async def serve(engine: Engine, uds: str | None, host: str, port: int, ready_cb=None) -> None:
    engine.loop = asyncio.get_running_loop()
    if uds:
        try:
            os.unlink(uds)
        except FileNotFoundError:
            pass
        srv = await asyncio.start_unix_server(engine.handle, path=uds, limit=1 << 22, backlog=1024)
        os.chmod(uds, 0o666)
    else:
        srv = await asyncio.start_server(engine.handle, host=host, port=port, limit=1 << 22, backlog=1024)
    if ready_cb:
        ready_cb()
    async with srv:
        await srv.serve_forever()
