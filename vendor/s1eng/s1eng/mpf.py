"""MPF (ruj, S1ENG_MPF=N): the HTTP front-end in N worker processes, so request handling gets N GILs.

#131 serves every request on one thread of the GPU process (server.serve: one asyncio loop runs Engine.handle /
decide for all connections). Under gVisor (the official runtime) every syscall, futex hand-off and epoll wait of that
thread is a Sentry round trip; in a returning wave (6 answers out, 5 requests in) it is 90-92 % busy (gvh: 183 -> 345
us of work per request), so a freed lane waits for the clients' next requests behind that one thread's serial work.

With S1ENG_MPF=N (1..8; 0 / unset: main.py never imports this module and the engine is #131):
  * start() (top of main(), before torch, CUDA or any thread exists) forks N workers. Each builds the tokenizer as
    main() does (prompt.Encoder; with S1ENG_TOK_TAIL=1 enable_tail(template_prefix())), runs server.Engine on its own
    asyncio loop (handle, decide, _answer, _send unchanged: every answer byte comes from that code) and gets its
    connections from the GPU process as file descriptors (SCM_RIGHTS on a socketpair). Its Engine's gpu is a _Remote:
    _run writes (job id, ARRIVE flag, the int64 token ids) to the worker's job pipe; the results come back on its
    result pipe and resolve the decide futures on the worker's loop (server._resolve, no extra self-pipe write).
  * The GPU process keeps the model, the graphs, the BatchLoop (pick_variant, ARRIVE's back counter, the hold rule,
    unchanged), the completers and the lean launch. Its loop class (loop_class(): a BatchLoop subclass) changes two
    things: `arriving` = the workers' arrival counters (a MAP_SHARED page, one writer each) minus the jobs already
    queued; and after each launch the batch's jobs get a shared record per worker, so the completer's unchanged
    per-job `j.done(res[r])` calls end in ONE write per worker per batch (the float32 rows of the bf16 outputs, as
    raw bytes; a launch exception travels as its kind - ValueError or other - and its message, so decide answers 400
    with the same text or 500 as #131 does). One receiver thread per worker turns each job record into a model.Job
    and calls BatchLoop.submit, as the HTTP thread did.
  * gate() (after TOK_TAIL, before the BatchLoop exists): every worker reports a digest of its tokenizer state (TOK_TAIL
    on / why off, template prefix ids, letter slots, the ids of every prompt.tail_samples() prompt) and of its Engine
    (temperature, token, input limit, health body); any worker missing or different -> "lever not used: mpf (...)",
    the workers are killed and #131's single-process server runs.
  * serve() (where main() would start the HTTP thread): the listening unix socket exactly as server.serve makes it
    (unlink, bind, backlog 1024, mode 0666) and an acceptor thread that hands each new connection to the live worker
    with the fewest open connections. A worker that dies is logged and reaped; the kernel closes its connections
    (the client reconnects), new ones go to the others, none left = refused. Any failure here before serving ->
    "lever not used: mpf (...)" and #131's server thread on the same loop object (which then behaves as BatchLoop).

FTOK (kai, ftok.py; S1ENG_FTOK / ARMS key ftok): each worker runs FTOK's start-up check after its tail encode when
env or an arm asks for it and reports the outcome (report field "ftok"; the digest is unchanged: FTOK's ids are the
tokenizer's); the GPU process logs it (ftok.merge_workers). An ARMS switch of ftok reaches every live worker as a
control record on its result pipe (Front.ctl_all: job id -1, kind K_CTL, JSON {"ftok": 0 / 1}), applied between two
requests on the worker's loop (worker_ctl).

Only ids and results cross the processes: the worker's ids are the GPU process's ids (same files, same code, and
the start-up gate), the GPU process runs the unchanged BatchLoop and graphs on them and returns the same float32
values; everything else is scheduling.
"""
from __future__ import annotations

from .server import OPTION_COUNT

import ctypes
import hashlib
import json
import mmap
import os
import select
import signal
import socket
import struct
import sys
import threading
import time

MAXW = 8
JH = struct.Struct("<qII")   # worker -> GPU: job id, flags, n (then n int64 token ids, or n bytes of JSON if F_CTL)
RH = struct.Struct("<qII")   # GPU -> worker: job id, kind, n (then n bytes)
F_ARR = 1                    # the request was counted as arriving (ARRIVE)
F_CTL = 1 << 31              # a control record (JSON), not a job
K_RES, K_VERR, K_ERR, K_CTL = 0, 1, 2, 3   # float32 row / ValueError(msg) / other exception / control (JSON)
OFF_ARR, OFF_CLOSED = 0, MAXW               # int64 slots of the shared page (worker k writes slot OFF_* + k)
GATE_S = float(os.environ.get("S1ENG_MPF_GATE_S", "300"))
PR_SET_PDEATHSIG = 1


def wanted(env=os.environ) -> int:
    v = env.get("S1ENG_MPF", "").strip()
    if v in ("", "0"):
        return 0
    n = int(v)
    if not 1 <= n <= MAXW:
        raise ValueError(f"S1ENG_MPF {n} not in 1..{MAXW}")
    return n


def _write_all(fd: int, data) -> None:
    mv = memoryview(data)
    while mv:
        n = os.write(fd, mv)
        mv = mv[n:]


def _ctl_record(obj) -> bytes:
    b = json.dumps(obj, separators=(",", ":")).encode()
    return JH.pack(-1, F_CTL, len(b)) + b


def digest(enc, engine) -> tuple[str, int]:
    """sha256 of everything a worker's answers depend on besides the GPU results: the tokenizer path (TOK_TAIL state,
    template prefix ids, letter slots, the ids of every tail_samples() prompt through enc.encode) and the Engine's
    constants. Computed the same way in the GPU process and in every worker."""
    from .prompt import tail_samples

    h = hashlib.sha256()

    def put(x):
        h.update(json.dumps(x, separators=(",", ":"), ensure_ascii=True).encode())
        h.update(b"\n")
    texts = tail_samples()
    put(["tail", getattr(enc, "_tail", None) is not None, list(getattr(enc, "_pre_ids", []) or [])])
    put(["prefix", list(enc.template_prefix())])
    put(["slots", list(enc.slots)])
    for t in texts:
        put(list(enc.encode(t)))
    put(["engine", repr(engine.temperature), hashlib.sha256(engine.token.encode()).hexdigest(), engine.max_tokens,
         engine.health_body.decode("utf-8")])
    return h.hexdigest(), len(texts)


class _W:
    __slots__ = ("k", "jr", "jw", "rr", "rw", "a", "b", "pid", "dead", "lock", "why")

    def __init__(self, k):
        self.k = k
        self.jr, self.jw = os.pipe()     # job pipe: worker writes, GPU process reads
        self.rr, self.rw = os.pipe()     # result pipe: GPU process writes, worker reads
        self.a, self.b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)   # connection fds, GPU -> worker
        self.pid = 0
        self.dead = False
        self.lock = threading.Lock()
        self.why = ""


class _Done:
    """A queued job's done(res) in the GPU process (called by the completer, or by run() / the receiver with the
    exception): the result record goes to the worker, alone or - when the batch record b is set - in one write with
    the batch's other results for the same worker."""
    __slots__ = ("f", "k", "jid", "b")

    def __init__(self, f, k, jid):
        self.f, self.k, self.jid, self.b = f, k, jid, None

    def __call__(self, res) -> None:
        try:
            if isinstance(res, BaseException):
                m = str(res).encode("utf-8", "surrogatepass")
                rec = RH.pack(self.jid, K_VERR if isinstance(res, ValueError) else K_ERR, len(m)) + m
            else:
                p = res.tobytes()
                rec = RH.pack(self.jid, K_RES, len(p)) + p
        except Exception as exc:  # noqa: BLE001 - e.g. an exception whose str() fails: decide answers 500, as #131's
            m = f"{type(exc).__name__}".encode()
            rec = RH.pack(self.jid, K_ERR, len(m)) + m
        try:
            b = self.b
            if b is None:
                self.f.write_res(self.k, rec)
                return
            b[1].append(rec)
            b[0] -= 1
            if b[0] <= 0:
                self.f.write_res(self.k, b"".join(b[1]))
        except Exception as exc:  # noqa: BLE001 - the completer thread must never die here
            self.f.note(f"result of job {self.jid} for worker {self.k} not sent: {type(exc).__name__}: {exc}")


def _make_loop(BatchLoop):
    class MPFLoop(BatchLoop):
        """BatchLoop with ARRIVE's `arriving` read from the workers and one result write per worker per batch.
        Unattached (_mpf None, e.g. after a refusal at serve()) it behaves as BatchLoop."""
        _arr_local = 0     # written by a local server.Engine (only after a refusal: #131's server on this loop)
        _mpf = None        # the Front, once serving

        @property
        def arriving(self):
            f = self._mpf
            return self._arr_local if f is None else self._arr_local + f.arriving()

        @arriving.setter
        def arriving(self, v):
            self._arr_local = v

        def _launch(self, ln, L, jobs, slot, pk):
            super()._launch(ln, L, jobs, slot, pk)
            f = self._mpf
            if f is not None:   # after the launch call (the graph is queued), before the completer can see the batch
                f.tag(jobs)
    return MPFLoop


class Front:
    """The GPU-process side: the workers, the gate, the receivers, the acceptor."""

    def __init__(self, n: int, log):
        self.n = n
        self.log = log
        self.mm = mmap.mmap(-1, mmap.PAGESIZE)     # MAP_SHARED | MAP_ANONYMOUS: shared with the forked workers
        self.arr = (ctypes.c_int64 * (2 * MAXW)).from_buffer(self.mm)
        self.w = [_W(k) for k in range(n)]
        self.live = tuple(range(n))
        self.subs = [0] * n           # jobs of worker k that were counted as arriving and are queued (receiver k only)
        self.handed = [0] * n         # connections handed to worker k (acceptor only)
        self.rr = 0
        self.lock = threading.Lock()
        self.gpu = None
        self.lsock = None
        self.gated = False
        self.Loop = None
        self.notes = 0
        self.parent = os.getpid()
        self.ws_closed = False
        self.threads = False   # receivers / acceptor started (their fds then stay open until the process exits)
        self.ftok_reports = {}   # FTOK: worker k -> "" (used), "off" (not asked) or why not, from the start-up reports

    # ------------------------------------------------------------------------------------------- start (fork)
    def fork_all(self, model_dir: str, manifest: dict, token: str, temperature: float, env) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        for w in self.w:
            pid = os.fork()
            if pid == 0:   # worker k: single-threaded child of a process with no thread but its main one
                code = 1
                try:
                    self._child_fds(w.k)
                    _worker(w.k, w, self.arr, self.parent, model_dir, manifest, token, temperature, env, self.log)
                    code = 0
                except BaseException as exc:  # noqa: BLE001
                    try:
                        self.log(f"mpf worker {w.k}: {type(exc).__name__}: {exc}")
                        _write_all(w.jw, _ctl_record({"k": w.k, "error": f"{type(exc).__name__}: {exc}"[:300]}))
                    except BaseException:  # noqa: BLE001
                        pass
                finally:
                    os._exit(code)
            w.pid = pid
        self._close_worker_ends()
        self.log(f"mpf: {self.n} worker processes forked (pids {[w.pid for w in self.w]}) before torch / CUDA")

    def _close_worker_ends(self) -> None:
        """The parent keeps the GPU-side ends only (EOF on a pipe then means its worker is gone)."""
        if self.ws_closed:
            return
        self.ws_closed = True
        for w in self.w:
            os.close(w.jw)
            os.close(w.rr)
            w.b.close()

    def _child_fds(self, k: int) -> None:
        for w in self.w:
            os.close(w.jr)
            os.close(w.rw)
            w.a.close()
            if w.k != k:
                os.close(w.jw)
                os.close(w.rr)
                w.b.close()

    # ------------------------------------------------------------------------------------------- gate
    def gate(self, enc, temperature: float, token: str, manifest: dict) -> str:
        """'' when every worker reported the GPU process's own digest; else why not (the caller logs and shuts down)."""
        try:
            return self._gate(enc, temperature, token, manifest)
        except Exception as exc:  # noqa: BLE001
            return f"gate: {type(exc).__name__}: {exc}"

    def _gate(self, enc, temperature, token, manifest) -> str:
        from .server import Engine

        e = Engine(enc, None, temperature, token, manifest["limits"]["max_input_tokens"],
                   manifest["weights"]["revision"])
        want, nt = digest(enc, e)
        tail_on = getattr(enc, "_tail", None) is not None
        t_end = time.monotonic() + GATE_S
        reps = []
        for w in self.w:
            rep = self._read_report(w, t_end)
            if rep is None:
                return f"worker {w.k} sent no start-up report within {GATE_S:.0f} s"
            if rep.get("error"):
                return f"worker {w.k} failed: {rep['error']}"
            if rep.get("digest") != want:
                return (f"worker {w.k} tokenizer / engine state differs from this process's (worker tail encode "
                        f"{rep.get('tail')!r}, here {'on' if tail_on else 'off'})")
            reps.append(rep)
        self.ftok_reports = {w.k: str(r.get("ftok", "off")) for w, r in zip(self.w, reps)}
        self.gated = True
        self.log(f"lever used: mpf gate: {self.n} workers, tokenizer + engine digest equal on {nt} prompts "
                 f"({want[:12]}), 0 rejected; worker start-up {max(r.get('t', 0) for r in reps):.1f} s")
        return ""

    def _read_report(self, w, t_end):
        buf = b""
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                return None
            r, _, _ = select.select([w.jr], [], [], left)
            if not r:
                return None
            d = os.read(w.jr, 65536)
            if not d:
                return {"error": "exited before reporting"}
            buf += d
            if len(buf) >= JH.size:
                jid, fl, n = JH.unpack_from(buf, 0)
                if not fl & F_CTL:
                    return {"error": "a job record before the start-up report"}
                if len(buf) >= JH.size + n:
                    if len(buf) > JH.size + n:
                        return {"error": "extra bytes after the start-up report"}
                    return json.loads(buf[JH.size:JH.size + n])

    def loop_class(self, BatchLoop):
        """The class main() builds its batch loop from: MPFLoop, or BatchLoop itself if that fails (serve() refuses)."""
        try:
            self.Loop = _make_loop(BatchLoop)
            return self.Loop
        except Exception as exc:  # noqa: BLE001
            self.log(f"mpf: loop class: {type(exc).__name__}: {exc}")
            return BatchLoop

    # ------------------------------------------------------------------------------------------- serve
    def serve(self, gpu, uds: str | None, t0: float) -> str:
        """'' once the workers serve (the caller then runs gpu.run()); else why not, with nothing left running."""
        try:
            return self._serve(gpu, uds, t0)
        except Exception as exc:  # noqa: BLE001
            g = self.gpu
            if g is not None:
                g._mpf = None
            return f"serve: {type(exc).__name__}: {exc}"

    def _serve(self, gpu, uds, t0) -> str:
        if not self.gated:
            return "not gated"
        if self.Loop is None or type(gpu) is not self.Loop:
            return "no batch lanes (the per-decision loop serves alone)"
        if not uds:
            return "no RUNTIME_UDS (TCP serving stays in one process)"
        ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:   # as server.serve's start_unix_server: unlink, bind, listen(backlog 1024), mode 0666
            try:
                os.unlink(uds)
            except FileNotFoundError:
                pass
            ls.bind(uds)
            ls.listen(1024)
            os.chmod(uds, 0o666)
        except Exception as exc:  # noqa: BLE001
            ls.close()
            return f"listening socket: {type(exc).__name__}: {exc}"
        self.lsock = ls
        for w in self.w:
            os.set_blocking(w.rw, False)
        self.gpu = gpu
        gpu._mpf = self
        if getattr(gpu, "ftok_ok", False):   # FTOK: an ARMS switch made before serving reaches the workers too
            with gpu.cv:                       # (_apply_arm sends under cv: no older value can overtake this one)
                self.ctl_all({"ftok": int(gpu.ftok)})
        self.threads = True
        try:
            for w in self.w:
                threading.Thread(target=self._rx, args=(w.k,), name=f"mpf-rx{w.k}", daemon=True).start()
            threading.Thread(target=self._accept, name="mpf-accept", daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            gpu._mpf = None
            self.gpu = None
            ls.close()
            return f"threads: {type(exc).__name__}: {exc}"
        self.log(f"serving on {uds} at {time.time() - t0:.1f}s")
        self.log(f"MPF on: {self.n} worker processes (pids {[w.pid for w in self.w]}) serve HTTP; this process "
                 f"accepts and hands connections to the least-loaded worker")
        return ""

    def shutdown(self) -> None:
        """Kill and reap the workers (our own children, by pid) and close our ends; nothing else is touched."""
        g = self.gpu
        if g is not None:
            g._mpf = None
        for w in self.w:
            w.dead = True
            if w.pid:
                try:
                    os.kill(w.pid, signal.SIGKILL)
                    os.waitpid(w.pid, 0)
                except OSError:
                    pass
        self.live = ()
        self._close_worker_ends()
        if self.threads:   # a receiver may still be in os.read on its pipe: never close (and recycle) its fd
            return
        for w in self.w:
            for fd in (w.jr, w.rw):
                try:
                    os.close(fd)
                except OSError:
                    pass
            w.a.close()

    # ------------------------------------------------------------------------------------------- GPU-process paths
    def arriving(self) -> int:
        a, s = self.arr, self.subs
        return sum(a[OFF_ARR + k] - s[k] for k in self.live)

    def tag(self, jobs) -> None:
        bt = None
        for j in jobs:
            d = j.done
            if type(d) is _Done:
                if bt is None:
                    bt = {}
                e = bt.get(d.k)
                if e is None:
                    e = bt[d.k] = [0, []]
                e[0] += 1
                d.b = e

    def write_res(self, k: int, data: bytes) -> None:
        w = self.w[k]
        if w.dead:
            return
        with w.lock:
            mv = memoryview(data)
            try:
                while mv:
                    try:
                        n = os.write(w.rw, mv)
                    except BlockingIOError:   # the worker is 64 KB behind: wait for it a little, then give up on it
                        if not select.select([], [w.rw], [], 1.0)[1]:
                            raise TimeoutError("result pipe full for 1 s") from None
                        continue
                    mv = mv[n:]
            except OSError as exc:
                self._dead(k, f"result pipe: {type(exc).__name__}: {exc}")

    def ctl_all(self, obj) -> None:
        """A control record (job id -1, kind K_CTL, JSON obj) to every live worker, on its result pipe (FTOK's ARMS
        switch: worker_ctl applies it on the worker's loop, between two requests)."""
        b = json.dumps(obj, separators=(",", ":")).encode()
        rec = RH.pack(-1, K_CTL, len(b)) + b
        for k in self.live:
            self.write_res(k, rec)

    def note(self, msg: str) -> None:
        self.notes += 1
        if self.notes <= 20:
            self.log(f"mpf: {msg}")

    def _dead(self, k: int, why: str) -> None:
        w = self.w[k]
        with self.lock:
            if w.dead:
                return
            w.dead = True
            w.why = why
            self.live = tuple(i for i in self.live if i != k)
        st = ""
        try:
            os.kill(w.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            _, status = os.waitpid(w.pid, 0)
            st = f", wait status {status}"
        except OSError:
            pass
        self.log(f"mpf: worker {k} (pid {w.pid}) is gone ({why}{st}); its connections are closed, new ones go to "
                 f"the {len(self.live)} other worker(s)" + ("" if self.live else ": NONE LEFT, new connections are refused"))

    def _rx(self, k: int) -> None:
        import numpy as np

        from .model import Job

        w = self.w[k]
        gpu = self.gpu
        subs = self.subs
        jsz = JH.size
        buf = bytearray()
        while True:
            try:
                d = os.read(w.jr, 1 << 16)
            except InterruptedError:
                continue
            except OSError:
                d = b""
            if not d:
                self._dead(k, "job pipe closed")
                return
            buf += d
            off, nb = 0, len(buf)
            while nb - off >= jsz:
                jid, fl, n = JH.unpack_from(buf, off)
                need = jsz + (n if fl & F_CTL else 8 * n)
                if nb - off < need:
                    break
                if fl & F_CTL:
                    self._ctl(k, bytes(buf[off + jsz:off + need]))
                else:
                    job = Job(np.frombuffer(buf, dtype=np.int64, count=n, offset=off + jsz).copy(), _Done(self, k, jid))
                    job.option_count = (fl >> 8) & 31
                    assert 1 <= job.option_count <= 16
                    if fl & F_ARR:   # ARRIVE: queued now, no longer arriving (the HTTP thread's decrement in #131)
                        subs[k] += 1
                    try:
                        gpu.submit(job)
                    except Exception as exc:  # noqa: BLE001 - as in #131, the submit's exception reaches decide
                        job.done(exc)
                off += need
            if off:
                del buf[:off]

    def _ctl(self, k: int, payload: bytes) -> None:
        try:
            msg = json.loads(payload)
        except ValueError:
            return
        if msg.get("error"):
            self.note(f"worker {k}: {msg['error']}")

    def _pick(self):
        live = self.live
        if not live:
            return None
        a, h = self.arr, self.handed
        r = self.rr
        self.rr = r + 1
        best = bo = None
        for i in range(len(live)):
            k = live[(r + i) % len(live)]
            o = h[k] - a[OFF_CLOSED + k]
            if bo is None or o < bo:
                best, bo = k, o
        return best

    def _accept(self) -> None:
        ls = self.lsock
        busy_err = 0
        while True:
            try:
                c, _ = ls.accept()
            except InterruptedError:
                continue
            except OSError as exc:
                if ls.fileno() < 0:
                    return
                busy_err += 1
                if busy_err <= 5:
                    self.log(f"mpf: accept: {type(exc).__name__}: {exc}")
                time.sleep(0.01)
                continue
            try:
                while True:
                    k = self._pick()
                    if k is None:
                        break   # no live worker: the connection is closed (refused)
                    try:
                        socket.send_fds(self.w[k].a, [b"c"], [c.fileno()])
                        self.handed[k] += 1
                        break
                    except OSError as exc:
                        self._dead(k, f"connection channel: {type(exc).__name__}: {exc}")
            finally:
                c.close()


def start(model_dir: str, manifest: dict, token: str, temperature: float, log, env=os.environ) -> Front:
    """main() top, before torch: fork the workers (raises when MPF cannot start; the caller logs it)."""
    n = wanted(env)
    if threading.active_count() != 1 or len(os.listdir("/proc/self/task")) != 1:
        raise RuntimeError("threads exist before the fork")
    f = Front(n, log)
    try:
        f.fork_all(model_dir, manifest, token, temperature, env)
    except BaseException:
        f.shutdown()
        raise
    return f


# ================================================================================================ worker process
def _worker_ftok(enc, env, k: int, log) -> str:
    """FTOK in worker k after its tail encode: '' = used, 'off' = not asked for (ftok is then never imported), else
    why not (the tail encode stays tokenizers')."""
    if env.get("S1ENG_FTOK", "") in ("", "0") and not env.get("S1ENG_ARMS"):
        return "off"
    from . import ftok

    if not ftok.wanted(env)[1]:
        return "off"
    log(f"mpf worker {k}: {ftok.setup(enc, env)}")
    return "" if ftok.OK else (ftok.WHY or "failed")


def worker_ctl(k: int, payload: bytes, log) -> None:
    """A control record from the GPU process (Front.ctl_all), on worker k's loop. {"ftok": 0 / 1}: FTOK's switch."""
    try:
        msg = json.loads(payload)
    except ValueError:
        return
    if not isinstance(msg, dict) or "ftok" not in msg:
        return
    ftm = sys.modules.get(f"{__package__}.ftok")
    if ftm is None or not ftm.OK:
        log(f"mpf worker {k}: ftok {msg['ftok']!r} not applied (start-up check failed or not run)")
        return
    prev = ftm.ACTIVE
    try:
        ftm.set_active(msg["ftok"])
    except (TypeError, ValueError) as exc:
        log(f"mpf worker {k}: ftok {msg['ftok']!r} not applied ({exc})")
        return
    if ftm.ACTIVE != prev:
        log(f"mpf worker {k}: ftok {int(prev)} -> {int(ftm.ACTIVE)}")


def _worker(k: int, w: _W, arr, parent: int, model_dir: str, manifest: dict, token: str, temperature: float, env,
            log) -> None:
    try:   # die with the GPU process
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass
    if os.getppid() != parent:
        return
    t0 = time.time()
    import asyncio

    import numpy as np

    from .prompt import Encoder
    from .server import Engine, _resolve

    enc = Encoder(model_dir)
    tail_why = "off"
    if env.get("S1ENG_TOK_TAIL", "0") == "1":   # as main.py (there prefix_ids = template_prefix(), or [] -> the same)
        tail_why = enc.enable_tail(enc.template_prefix())
    ftok_why = _worker_ftok(enc, env, k, log)   # FTOK (kai): after the tail encode, before the digest
    jw, rr = w.jw, w.rr
    cs = w.b
    pending: dict = {}
    seq = [0]

    class Remote:
        """server.Engine's gpu in a worker: ARRIVE's counter is this worker's slot of the shared page (only this
        thread writes it; the GPU process subtracts the jobs it has queued)."""
        arrive = env.get("S1ENG_ARRIVE", "1") == "1"   # = BatchLoop.arrive (the GPU process's default)

        @property
        def arriving(self):
            return arr[OFF_ARR + k]

        @arriving.setter
        def arriving(self, v):
            arr[OFF_ARR + k] = v

    remote = Remote()

    class WEngine(Engine):
        def _run(self, ids, tm=None):
            # server.Engine._run with the GPU hand-off through the job pipe: the future resolves when the GPU
            # process's result record for this job id arrives (on_res below)
            fut = self.loop.create_future()
            fl = F_ARR if (tm is not None and tm.pop("arr", False)) else 0
            count = OPTION_COUNT.get()
            assert 1 <= count <= 16
            fl |= count << 8
            a = np.asarray(ids, dtype=np.int64)
            seq[0] += 1
            jid = seq[0]
            pending[jid] = fut
            try:
                _write_all(jw, JH.pack(jid, fl, len(a)) + a.tobytes())
            except BaseException:
                pending.pop(jid, None)
                if fl & F_ARR:   # never reached the GPU process: no longer arriving
                    remote.arriving -= 1
                raise
            return fut

    engine = WEngine(enc, remote, temperature, token, manifest["limits"]["max_input_tokens"],
                     manifest["weights"]["revision"])
    enc.encode("warm-up")
    dg, nt = digest(enc, engine)
    _write_all(jw, _ctl_record({"k": k, "pid": os.getpid(), "tail": tail_why, "digest": dg, "n": nt,
                                "ftok": ftok_why, "t": round(time.time() - t0, 2)}))
    log(f"mpf worker {k} (pid {os.getpid()}): tokenizer + engine ready in {time.time() - t0:.1f}s "
        f"(tail encode {'on' if not tail_why else tail_why})")

    async def main():
        loop = asyncio.get_running_loop()
        engine.loop = loop
        buf = bytearray()
        rsz = RH.size

        def gone(why):
            log(f"mpf worker {k}: the GPU process is gone ({why}): exiting")
            os._exit(0)

        def on_res():
            try:
                d = os.read(rr, 1 << 16)
            except (BlockingIOError, InterruptedError):
                return
            if not d:
                gone("result pipe closed")
            buf.extend(d)
            off, nb = 0, len(buf)
            while nb - off >= rsz:
                jid, kind, n = RH.unpack_from(buf, off)
                if nb - off - rsz < n:
                    break
                p = bytes(buf[off + rsz:off + rsz + n])
                off += rsz + n
                if kind == K_CTL:   # a control record (Front.ctl_all), not a job's result
                    worker_ctl(k, p, log)
                    continue
                fut = pending.pop(jid, None)
                if fut is None:
                    continue
                if kind == K_RES:
                    res = np.frombuffer(p, dtype=np.float32).copy()
                elif kind == K_VERR:
                    res = ValueError(p.decode("utf-8", "surrogatepass"))
                else:
                    res = RuntimeError(p.decode("utf-8", "surrogatepass"))
                _resolve(fut, res)
            if off:
                del buf[:off]

        async def handle(reader, writer):
            try:
                await engine.handle(reader, writer)
            finally:
                arr[OFF_CLOSED + k] += 1

        async def conn(fd):
            try:
                sock = socket.socket(fileno=fd)
            except OSError:
                os.close(fd)
                arr[OFF_CLOSED + k] += 1
                return
            try:   # as start_unix_server's accept path: StreamReader(limit 4 MiB) + StreamReaderProtocol(handle)
                reader = asyncio.StreamReader(limit=1 << 22, loop=loop)
                proto = asyncio.StreamReaderProtocol(reader, handle, loop=loop)
                await loop.connect_accepted_socket(lambda: proto, sock)
            except Exception:  # noqa: BLE001
                sock.close()
                arr[OFF_CLOSED + k] += 1

        def on_fd():
            try:
                msg, fds, _, _ = socket.recv_fds(cs, 256, 64)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as exc:
                gone(f"connection channel: {exc}")
            if not msg and not fds:
                gone("connection channel closed")
            for fd in fds:
                loop.create_task(conn(fd))

        os.set_blocking(rr, False)
        cs.setblocking(False)
        loop.add_reader(rr, on_res)
        loop.add_reader(cs.fileno(), on_fd)
        await loop.create_future()   # forever

    asyncio.run(main())
