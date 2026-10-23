"""gvhb: compare two exactness captures (capture2.py output, same seed) pass by pass: every answer byte-equal (the
canonical JSON of answers[<qid>]) and no client failure.   python3 -I cmpcap2.py A.json B.json  (exit 1 on a difference)"""
import json
import sys

A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = A.get("seed") == B.get("seed") and A.get("idents") == B.get("idents")
if not ok:
    print("different item sets")
for name, ans in A["passes"].items():
    bans = B["passes"].get(name, {})
    eq = sum(1 for k, v in ans.items() if bans.get(k) == v)
    fa, fb = len(A["fails"].get(name, [])), len(B["fails"].get(name, []))
    print(f"{name}: {eq}/{len(ans)} byte-equal (B has {len(bans)}; client fails A {fa} B {fb})")
    ok &= eq == len(ans) == len(bans) and fa == 0 and fb == 0
print("ALL BYTE-EQUAL" if ok else "DIFFERENCES FOUND")
sys.exit(0 if ok else 1)
