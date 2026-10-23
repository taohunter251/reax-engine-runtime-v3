# MPF: multi-process HTTP front-end (ruj), `S1ENG_MPF=N`

In this image (pv131mpf): pv131, the provable #131 (every cubin next to its `.cu`, build lines in
`s1eng/KERNELS.md`, GRP off), + MPF with `S1ENG_MPF=4`. MPF is Python only (`s1eng/mpf.py` + three hooks in
`s1eng/main.py` = `mpf_vs_131.patch`; pv131's `main.py` is #131's) and adds no compiled file. This `mpf.py` is the
gvhb m2 version: m1 (measured on #131 below, dashboard #134) + one exception path (`_Done.__call__`: an exception
whose `str()` fails is answered 500, as #131 would, instead of being dropped). The dashboard result and later notes
are in the ruj lever folder `levers_for_rgh/gvh/`; the pv131mpf results in `build/pv131mpf/PROGRESS.md`.

## Result on #131 (m1, Verda RTX PRO 6000 dev VM, gVisor runsc, official regime)

Separate plain starts in ABBA order vs #131 (1A 2B 3B 4A 5A 6B 7B 8A), 30 blocks per start of one 256-item c16
timed run after 3.0 s idle, the same items in every start: **+0.933 %** (start means, se 0.070, 95 % CI
+0.761..+1.105, df 6); pairs +0.91 / +0.83 / +1.00 / +1.00 %; MPF faster in 110 / 120 item-matched blocks; energy
per run -0.70 % (CI -1.08..-0.32); c1 p50 unchanged (8.44-8.49 ms vs 8.42-8.49 ms); serving 2.4 s later (152.4 vs
150.0 s). All 4 MPF starts exact: c16 2000 + c16r 2000 + c1 128 byte-equal to the reference capture, probe 58/58,
every #131 start-up gate as in #131, MPF gate 4 workers / 0 rejected.

## What it does

#131 serves every request on one thread of the GPU process (one asyncio loop runs `server.Engine.handle` / `decide`
for all connections). Under gVisor, the official runtime, every syscall, futex hand-off and epoll wait of that thread
is a Sentry round trip. In a returning wave (6 answers out, 5 requests in) the thread is 90-92 % busy, and its work
per request is 1.9x slower than under runc (183 -> 345 us). A freed lane waits for the clients' next requests behind
that one thread's serial work (the gvh breakdown: 87 % of runsc's lane-turnaround excess).

With `S1ENG_MPF=N` (1..8):

* `main()` forks N worker processes at its top, before torch, CUDA or any thread exists (`mpf.start`).
* Each worker builds the tokenizer as `main()` does (`prompt.Encoder`; with `S1ENG_TOK_TAIL=1`,
  `enable_tail(template_prefix())`) and runs `server.Engine` on its own asyncio loop: `handle`, `decide`, `_answer`
  and `_send` are unchanged. Its `gpu` is a remote stub: `_run` writes (job id, ARRIVE flag, the int64 token ids) to
  the worker's job pipe, and the result record resolves the decide future on the worker's loop.
* The GPU process keeps the model, the graphs, the BatchLoop (`pick_variant`, ARRIVE's back counter, the hold rule),
  the completers and the lean launch. Its loop class is a BatchLoop subclass that changes two things:
  * `arriving` = the workers' arrival counters (a MAP_SHARED page, one writer each) minus the jobs already queued;
  * after each launch, the batch's jobs get one shared record per worker, so the completer's unchanged per-job
    `j.done(res[r])` calls end in ONE pipe write per worker per batch.
* One receiver thread per worker turns each job record into a `model.Job` and calls `BatchLoop.submit`, as the HTTP
  thread did.
* The GPU process listens on `RUNTIME_UDS` exactly as `server.serve` does (unlink, bind, backlog 1024, mode 0666). An
  acceptor thread hands each new connection to the live worker with the fewest open connections (SCM_RIGHTS).

`S1ENG_MPF` unset or `0`: `main.py` never imports `mpf.py`, and the engine is #131 byte for byte (in this
image: pv131).

## Why the answers are the same

* Every answer byte comes from the unchanged `server.py` / `prompt.py` code in the worker, on the same tokenizer files.
* Start-up gate: every worker reports a sha256 of its tokenizer state (TOK_TAIL on / off, template prefix ids, letter
  slots, the ids of all 128 `tail_samples()` prompts) and its Engine constants (temperature, token, input limit, health
  body). Any worker missing or different: `lever not used: mpf (...)`, the workers are killed and #131's
  single-process server runs.
* The GPU process runs the unchanged BatchLoop and graphs on the same ids and returns the same values: the float32
  rows of the bf16 outputs, as raw bytes. A launch exception travels as its kind (ValueError or other) and its message,
  so `decide` answers 400 with the same text, or 500, as #131 does.
* Everything else is scheduling: when a request reaches the queue, and which connection a worker serves.

## Robustness

* Any setup failure before serving logs `lever not used: mpf (...)` and #131's server runs.
* A worker that dies is logged and reaped. The kernel closes its connections and the client reconnects. New
  connections go to the other workers; with none left they are refused.
* Workers die with the GPU process (PR_SET_PDEATHSIG, and EOF on their pipes).

## CPU harness (before the image)

`tools/vm/mpfh.py` runs `main.py`'s serving flow over a deterministic fake GPU in the #131 image. It compares MPF
with #131's server on the same fake, under runc and runsc, N = 2 and 4:

* c16 2000, c16r 2000 and c1 128 byte-equal, probe 58/58;
* 21 HTTP edge cases and a c1 pass of 300, byte-equal raw responses: health GET / HEAD / POST, 401, 400s, 404, 405s,
  chunked, HTTP/1.0, keep-alive, malformed line, malformed criteria (500), over-limit input;
* injected launch errors (ValueError -> 400 with its text, other -> 500), byte-equal;
* 20 timed runs per session with 0 failures, including one with a worker killed mid-run.
