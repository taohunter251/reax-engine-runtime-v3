"""skx6t unit test of the ENGINE code path (ruj): s1eng.mx from the skx6t tree (mounted over /opt/s1eng/s1eng) with
S1ENG_SKX6T set: lib() loads both cubins, tail_check passes at a few keys (and catches a broken variant), mx.run
switches to the variant only at gated keys and only while the switch is on (the FCAB noskt capture), CUDA-graph
capture of mx.run with the variant replays the same bytes for a new M1 (read from device memory).
  python -u unit_mx.py
"""
import os
import sys
import time

os.environ.setdefault("S1ENG_SKX6T", "1")
import torch  # noqa: E402

sys.path.insert(0, "/opt/s1eng")
from s1eng import mx  # noqa: E402

t0 = time.time()
L = mx.lib()
assert L is not None, "mx.lib() is None"
print("variant", mx.tail_mode(), "kernels", sorted(L.fn), "variants", len(L.tfn), flush=True)
g = torch.Generator(device="cuda").manual_seed(1)
ok = True
for (N, K, c1, c2, R) in ((2560, 9216, 2304, 1344, 832), (2560, 4096, 1024, 512, 768), (4096, 2560, 2560, 512, 640),
                          (4096, 2560, 2560, 512, 896), (2048, 2560, 448, 192, 704), (1024, 2560, 448, 192, 576)):
    W = (torch.randn((N, K), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    name = mx.kernel_for(N, K, R)
    for v in ("t", "m"):
        t1 = time.time()
        nd = mx.tail_check(name, W, c1, c2, R, v, g)
        print(f"tail_check {name} {N}x{K} R{R} {v}: {nd} differ ({time.time() - t1:.2f}s)", flush=True)
        ok &= nd == 0
    # the gate must catch a variant that is off by one ulp in one output value
    real = mx._launch

    def broken(L_, kname, W_, x_, out_, *a, **kw):
        real(L_, kname, W_, x_, out_, *a, **kw)
        if not kname.startswith("gemm_skx6_"):
            out_.view(torch.int16)[x_.shape[0] // 2, 3] += 1

    mx._launch = broken
    bad = mx.tail_check(name, W, c1, c2, R, "m", g)
    mx._launch = real
    print(f"  broken variant: {bad} differ (must be > 0)", flush=True)
    ok &= bad > 0
    # mx.run uses the variant only at a gated key and only with the switch on
    x = torch.randn((R, K), generator=g, device="cuda").to(torch.bfloat16)
    m1 = torch.tensor([R // 2 + 3], dtype=torch.int32, device="cuda")
    o_old = torch.empty((R, N), dtype=torch.bfloat16, device="cuda")
    mx.TAIL.clear()
    mx.run(name, W, x, o_old, c1, c2, m1=m1)
    mx.TAIL[(name, N, K, R)] = mx.tail_name(name, mx.tail_mode())
    o_new = torch.empty_like(o_old)
    mx.run(name, W, x, o_new, c1, c2, m1=m1)
    torch.cuda.synchronize()
    same = torch.equal(o_old.view(torch.int16), o_new.view(torch.int16))
    print(f"  mx.run old vs variant at M1 {R // 2 + 3}: {'equal' if same else 'DIFFER'}", flush=True)
    ok &= same
    # graph capture with the variant; replay after moving the boundary on the device
    gr = torch.cuda.CUDAGraph()
    o_g = torch.empty_like(o_old)
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        mx.run(name, W, x, o_g, c1, c2, m1=m1)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr, stream=st):
            mx.run(name, W, x, o_g, c1, c2, m1=m1)
    for mv in (0, 1, R // 3, R - 64, R):
        m1.fill_(mv)
        gr.replay()
        mx.TAIL.clear()
        o_r = torch.empty_like(o_old)
        mx.run(name, W, x, o_r, c1, c2, m1=m1)
        mx.TAIL[(name, N, K, R)] = mx.tail_name(name, mx.tail_mode())
        torch.cuda.synchronize()
        same = torch.equal(o_g.view(torch.int16), o_r.view(torch.int16))
        ok &= same
        if not same:
            print(f"  graph replay at M1 {mv}: DIFFER", flush=True)
    # switch off at call time (FCAB noskt capture): the gated key falls back to skx6
    used = []
    real = mx._launch
    mx._launch = lambda L_, kname, *a, **kw: (used.append(kname), real(L_, kname, *a, **kw))
    mx.run(name, W, x, o_r, c1, c2, m1=m1)
    os.environ["S1ENG_SKX6T"] = "0"
    mx.run(name, W, x, o_r, c1, c2, m1=m1)
    os.environ["S1ENG_SKX6T"] = "1"
    mx._launch = real
    print(f"  kernels on / off: {used}", flush=True)
    ok &= used == [mx.tail_name(name, mx.tail_mode()), name]
    mx.TAIL.clear()
print("UNIT", "PASS" if ok else "FAIL", f"({time.time() - t0:.1f}s)", flush=True)
