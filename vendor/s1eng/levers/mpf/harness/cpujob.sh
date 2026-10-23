#!/bin/bash
# gvhb CPU harness session (ruj; CPU ONLY: no --gpus; run under the GPU lock so no GPU measurement runs next to it):
# one container of the #131 image with the gvhb tree mounted over /opt/s1eng and the tools at /hb, running
# /hb/mpfh.py (main.py's serving flow over a deterministic fake GPU) with the validator-like flags; then from the host
# (client CPUs 28-29): capture2 (c16 N, c16r N, c1 C1) -> capture.json, the 58-input probe -> probe.json, edge.py
# (edge cases + c1 raw) -> edge.json, soak.py (timed runs; optional worker kill) -> soak.json; engine log.
#   cpujob.sh <tag> <runc|runsc> <mpf N|0> [-e K=V ...]     env: CAP_N (2000) C1 (128) SOAK (20) KILL (block, runc)
set -u
TAG=$1; RT=$2; NMPF=$3; shift 3
EXTRA=("$@")
IMG=${IMG:-127.0.0.1:5000/ruj/engine@sha256:e7c8807d44bb23eed48470e15803c34c858f31490ddcba98027bcb910238eda2}
H=/srv/reax/gvhb
ENG=${ENG:-0-1,22-27}; DRV=${DRV:-28-29}
CAP_N=${CAP_N:-2000}; C1=${C1:-128}; SOAK=${SOAK:-20}; KILL=${KILL:--1}
D=$H/runs/$TAG
rm -rf "$D"; mkdir -p "$D/io"; chmod 777 "$D/io"
python3 -c "import secrets;print(secrets.token_hex(24))" > "$D/io/token"; chmod 644 "$D/io/token"
NAME=ruj-gvhb-$TAG
docker rm -f "$NAME" >/dev/null 2>&1
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) start $TAG rt $RT mpf $NMPF eng $ENG ${EXTRA[*]}" | tee "$D/run.log"
docker run -d --runtime "$RT" --name "$NAME" --label reax.ours=1 --network none --read-only \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g --cap-drop ALL --security-opt no-new-privileges --pids-limit 1024 \
  --ulimit nofile=4096:4096 --memory 16g --memory-swap 16g --cpus 8 --cpuset-cpus "$(/srv/reax/bin/cpuset_req "$RT" "$ENG")" \
  --shm-size 2g --user 10002:10002 -v /srv/reax/models/plumb-4b:/model:ro -v "$D/io":/run/reax \
  -v $H/tree/s1eng:/opt/s1eng:ro -v $H/tools:/hb:ro \
  -e RUNTIME_UDS=/run/reax/engine.sock -e S1M_INTERNAL_TOKEN_FILE=/run/reax/token -e PLUMB_MODEL_DIR=/model \
  -e HF_HUB_OFFLINE=1 -e NVIDIA_VISIBLE_DEVICES=void -e S1ENG_MPF="$NMPF" -e HB_ERR="${HB_ERR:-0}" "${EXTRA[@]}" --entrypoint python "$IMG" \
  /hb/mpfh.py >> "$D/run.log" 2>&1
CID=$(docker inspect "$NAME" --format '{{.Id}}' 2>/dev/null)
CG=$(/srv/reax/bin/cgdir "$CID")
echo "cgroup $CG cpuset $(cat $CG/cpuset.cpus.effective 2>/dev/null)" >> "$D/run.log"
V=/srv/reax/venv/bin/python
T=/srv/reax/fixb/tools
python3 -I $T/probe.py --health "$D/io/engine.sock" 600 >> "$D/run.log" 2>&1 || echo "health timeout" | tee -a "$D/run.log"
cat "$CG/cpu.stat" > "$D/cpu0.stat" 2>/dev/null
RC=0
taskset -c "$DRV" $V $T/capture2.py --sock "$D/io/engine.sock" --token-file "$D/io/token" --out "$D/capture.json" \
  --seed 9001 --n "$CAP_N" --c1-n "$C1" >> "$D/run.log" 2>&1 || RC=1
taskset -c "$DRV" python3 -I $T/probe.py "$D/probe.json" "$TAG=$D/io/engine.sock:$D/io/token" > "$D/probe.log" 2>&1 || RC=1
taskset -c "$DRV" $V $H/tools/edge.py run --sock "$D/io/engine.sock" --token-file "$D/io/token" --out "$D/edge.json" \
  --c1 300 >> "$D/run.log" 2>&1 || RC=1
KP=0
if [ "$KILL" -ge 0 ] && [ "$RT" = runc ] && [ "$NMPF" != 0 ]; then   # an MPF worker's host pid (a child of the engine)
  EP=$(docker inspect "$NAME" --format '{{.State.Pid}}')
  KP=$(pgrep -P "$EP" | head -1)
  echo "kill test: engine pid $EP, worker pid $KP at block $KILL" | tee -a "$D/run.log"
fi
[ "$SOAK" -gt 0 ] && { taskset -c "$DRV" $V $H/tools/soak.py --sock "$D/io/engine.sock" --token-file "$D/io/token" \
  --out "$D/soak.json" --blocks "$SOAK" --gap 1.0 --kill-pid "$KP" --kill-block "$KILL" >> "$D/run.log" 2>&1 || RC=1; }
if [ "$KP" != 0 ]; then   # after the kill: new connections still served
  taskset -c "$DRV" python3 -I $T/probe.py "$D/probe2.json" "$TAG=$D/io/engine.sock:$D/io/token" > "$D/probe2.log" 2>&1 || RC=1
fi
cat "$CG/cpu.stat" > "$D/cpu1.stat" 2>/dev/null
cat "$CG/memory.peak" > "$D/memory.peak" 2>/dev/null
docker top "$NAME" -o pid,ppid,rss,etime,args > "$D/top.txt" 2>&1
docker logs "$NAME" > "$D/engine.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "$(date -u +%FT%T) done rc $RC" | tee -a "$D/run.log"
touch "$D/done"
exit $RC
