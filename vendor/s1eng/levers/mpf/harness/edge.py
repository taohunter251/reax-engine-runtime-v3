#!/usr/bin/env python3
"""gvhb: raw HTTP responses of one engine (unix socket) for edge cases and a c1 pass, saved byte for byte, to compare
MPF with #131's single-process server (status line, headers and body must be identical).

  edge.py run --sock S --token-file T --out O [--c1 300 --seed 9100]
  edge.py cmp A.json B.json            (exit 1 on any difference)
Cases: /health GET / HEAD / POST, decide GET, unknown path, bad / missing token, invalid JSON, two questions, bad type,
chunked body, HTTP/1.0, connection: close, two requests on one keep-alive connection, a malformed request line, a
5 MiB header (413), malformed criteria of each question kind (500), over-limit input (400); then --c1 decisions
(validator items, one at a time on one keep-alive connection) with their raw responses.
"""
import argparse
import json
import socket
import sys
import time

sys.path.insert(0, "/srv/reax/prof2/abl")


def recv_response(s, head_only=False):
    """One HTTP/1.1 response (content-length framing) from s; returns raw bytes."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(65536)
        if not d:
            return buf
        buf += d
    head, _, rest = buf.partition(b"\r\n\r\n")
    n = 0
    for line in head.split(b"\r\n")[1:]:
        k, _, v = line.partition(b":")
        if k.strip().lower() == b"content-length":
            n = int(v.strip())
    if head_only:
        n = 0
    while len(rest) < n:
        d = s.recv(65536)
        if not d:
            break
        rest += d
    return head + b"\r\n\r\n" + rest[:n]


def one(sock_path, raw: bytes, head_only=False, timeout=60.0, read_all=False):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall(raw)
    if read_all:   # the server closes after the response (or before: empty)
        buf = b""
        while True:
            try:
                d = s.recv(65536)
            except socket.timeout:
                break
            if not d:
                break
            buf += d
        s.close()
        return buf
    out = recv_response(s, head_only)
    s.close()
    return out


def req(method, path, body=b"", token=None, extra="", version="HTTP/1.1", ctype="application/json", close=True):
    h = f"{method} {path} {version}\r\nhost: engine\r\n"
    if ctype and body:
        h += f"content-type: {ctype}\r\n"
    if token is not None:
        h += f"x-s1m-internal-token: {token}\r\n"
    if body:
        h += f"content-length: {len(body)}\r\n"
    if close:
        h += "connection: close\r\n"
    return (h + extra + "\r\n").encode() + body


def decision_body(state, qid, q):
    return json.dumps({"request_id": "edge", "deadline_unix_ms": 0, "state": state, "questions": {qid: q}},
                      ensure_ascii=False).encode()


def run(a):
    tok = open(a.token_file).read().strip()
    S = a.sock
    good = decision_body({"ref": "0a1b2c3d", "notes": "the parcel arrived two days late"}, "q1",
                         {"type": "noul", "instructions": "Was the delivery late?"})
    cases = {}

    def case(name, raw, **kw):
        t = time.time()
        try:
            r = one(S, raw, **kw)
        except Exception as exc:  # noqa: BLE001
            r = f"EXC {type(exc).__name__}: {exc}".encode()
        cases[name] = {"raw": r.decode("latin-1"), "s": round(time.time() - t, 4)}
    case("health_get", req("GET", "/health"))
    case("health_head", req("HEAD", "/health"), head_only=True)
    case("health_post", req("POST", "/health", b"{}"))
    case("decide_get", req("GET", "/internal/v1/decide", token=tok))
    case("unknown", req("GET", "/nope"))
    case("bad_token", req("POST", "/internal/v1/decide", good, token="x" * 48))
    case("no_token", req("POST", "/internal/v1/decide", good))
    case("bad_json", req("POST", "/internal/v1/decide", b"{not json", token=tok))
    case("two_q", req("POST", "/internal/v1/decide", json.dumps({"state": 1, "questions": {
        "a": {"type": "noul"}, "b": {"type": "noul"}}}).encode(), token=tok))
    case("bad_type", req("POST", "/internal/v1/decide", decision_body(1, "q", {"type": "free"}), token=tok))
    case("good", req("POST", "/internal/v1/decide", good, token=tok))
    chunks = b"".join(b"%x\r\n" % len(good[i:i + 37]) + good[i:i + 37] + b"\r\n" for i in range(0, len(good), 37)) + b"0\r\n\r\n"
    case("chunked", (f"POST /internal/v1/decide HTTP/1.1\r\nhost: engine\r\nx-s1m-internal-token: {tok}\r\n"
                     f"transfer-encoding: chunked\r\nconnection: close\r\n\r\n").encode() + chunks)
    case("http10", req("POST", "/internal/v1/decide", good, token=tok, version="HTTP/1.0", close=False), read_all=True)
    case("keepalive2", req("POST", "/internal/v1/decide", good, token=tok, close=False)
         + req("GET", "/health", close=True), read_all=True)
    case("bad_line", b"GARBAGE\r\nhost: x\r\n\r\n", read_all=True)
    case("big_header", b"GET /health HTTP/1.1\r\nx-big: " + b"a" * (5 << 20) + b"\r\n\r\n", read_all=True)
    case("crit_choice_int", req("POST", "/internal/v1/decide", decision_body(
        "s", "q", {"type": "choice", "instructions": "x", "criteria": 5}), token=tok))
    case("crit_noul_list", req("POST", "/internal/v1/decide", decision_body(
        "s", "q", {"type": "noul", "instructions": "x", "criteria": ["a"]}), token=tok))
    case("crit_score_none", req("POST", "/internal/v1/decide", decision_body(
        "s", "q", {"type": "score", "instructions": "x"}), token=tok))
    case("crit_choice_list", req("POST", "/internal/v1/decide", decision_body(
        "s", "q", {"type": "choice", "instructions": "x", "criteria": ["up", "down", "flat"]}), token=tok))
    case("over_limit", req("POST", "/internal/v1/decide", decision_body(
        {"ref": "1", "notes": " ".join(["parcel delivery late"] * 2600)}, "q", {"type": "noul", "instructions": "x"}),
        token=tok))
    out = {"cases": cases, "c1": []}
    if a.c1:
        from driver2 import fixed_decisions
        decs = fixed_decisions(a.c1, a.seed)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(60)
        s.connect(S)
        for d in decs:
            body = decision_body(d.state, d.question.ident, d.question.author_dict())
            s.sendall(req("POST", "/internal/v1/decide", body, token=tok, close=False))
            r = recv_response(s)
            if not r:   # the server closed (an error status with close): reconnect
                s.close()
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(60)
                s.connect(S)
                s.sendall(req("POST", "/internal/v1/decide", body, token=tok, close=False))
                r = recv_response(s)
            out["c1"].append(r.decode("latin-1"))
        s.close()
    json.dump(out, open(a.out, "w"))
    st = {}
    for r in out["c1"]:
        code = r.split(" ", 2)[1] if r.startswith("HTTP/") else "?"
        st[code] = st.get(code, 0) + 1
    print(f"edge: {len(cases)} cases, c1 {len(out['c1'])} responses, status {st}")
    for k, v in cases.items():
        print(f"  {k:18s} {v['raw'][:60]!r}")


def cmp(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    bad = 0
    for k in A["cases"]:
        if A["cases"][k]["raw"] != B["cases"].get(k, {}).get("raw"):
            bad += 1
            print(f"case {k} DIFFERS:\n  A {A['cases'][k]['raw'][:300]!r}\n  B {B['cases'].get(k, {}).get('raw', '')[:300]!r}")
    if set(A["cases"]) != set(B["cases"]):
        bad += 1
        print("case sets differ")
    nd = sum(1 for x, y in zip(A["c1"], B["c1"]) if x != y) + abs(len(A["c1"]) - len(B["c1"]))
    bad += nd
    print(f"cases {len(A['cases'])}: {sum(1 for k in A['cases'] if A['cases'][k]['raw'] == B['cases'].get(k, {}).get('raw'))} "
          f"byte-equal; c1 {len(A['c1'])} vs {len(B['c1'])}: {nd} differ")
    print("ALL BYTE-EQUAL" if not bad else "DIFFERENCES FOUND")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--sock", required=True)
    r.add_argument("--token-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--c1", type=int, default=0)
    r.add_argument("--seed", type=int, default=9100)
    c = sp.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else cmp(a)
