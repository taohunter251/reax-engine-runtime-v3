#!/bin/sh
# Rebuild UID 56's two compiled kernels from the sources in this folder and compare them with the cubins in the committed image.
# usage: ./verify.sh <nvcc 13.4> <nvcc 13.0>      needs: crane, tar, sha256sum
set -e
NVCC134=${1:?path to nvcc 13.4}
NVCC130=${2:?path to nvcc 13.0}
HERE=$(cd "$(dirname "$0")" && pwd)
REF=ghcr.io/masterdev777/s1k@sha256:23dcdecc1519edf813699ab48adadd1df07a269016da3816bee37cba84e8a89d
W=$(mktemp -d)
for L in sha256:fb43b7b932e484f245df4afe2f0dc5fae6e57991a058efb85d787a545a8c2581 \
         sha256:c1e8a3847be63c78e85e860f4d7645688177055f671a40491de2b428ff803873; do
  crane blob "ghcr.io/masterdev777/s1k@$L" | tar -xz -C "$W" opt/s1eng/s1eng 2>/dev/null || true
done
cd "$W"
cp "$HERE/ngemm9b.cu" "$HERE/ffs.cu" .
build() {  # build <nvcc> <log> <args...>: keep ptxas's -v chatter out of sight, but show the log when the build fails
  nv=$1; log=$2; shift 2
  if ! "$nv" "$@" > "$log" 2>&1; then
    echo "build failed: $nv $*"; tail -20 "$log"
    echo "(CUDA 13.0's headers need a glibc <= 2.39 host, e.g. Ubuntu 24.04 with g++ 13)"; exit 1
  fi
}
build "$NVCC134" ngemm9b.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu
build "$NVCC130" ffs.log -arch=sm_120a -O3 -cubin -Xptxas -v -o ffs.cubin ffs.cu
for f in ngemm9b ffs; do
  a=$(sha256sum "$f.cubin" | cut -d' ' -f1)
  b=$(sha256sum "opt/s1eng/s1eng/$f.cubin" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "$f.cubin: byte-identical ($a)"; else echo "$f.cubin: DIFFERS (rebuilt $a, image $b)"; fi
done
echo "image: $REF"
rm -rf "$W"
