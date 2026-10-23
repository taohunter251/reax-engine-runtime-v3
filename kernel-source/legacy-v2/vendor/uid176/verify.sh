#!/bin/sh
# Rebuild UID 176's compiled kernels from the sources in this folder and compare them with the cubins in the committed image,
# including the older kkt2n.cubin that sits in a lower layer (layer 20) under the newer one (layer 24).
# usage: ./verify.sh <nvcc 13.4> <nvcc 13.0>      needs: crane, tar, sha256sum
set -e
NVCC134=${1:?path to nvcc 13.4}
NVCC130=${2:?path to nvcc 13.0}
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=ghcr.io/masterdev777/s1n
REF=$IMG@sha256:01e6e113da15bbc5f992779bb099a55d99b936ab088b64fb4967ef557f72db6d
W=$(mktemp -d)
layer() {  # layer <n> <digest>: unpack one layer into $W/L<n>
  mkdir -p "$W/L$1"; crane blob "$IMG@$2" | tar -xz -C "$W/L$1" opt/s1eng/s1eng 2>/dev/null || true
}
layer 16 sha256:fb43b7b932e484f245df4afe2f0dc5fae6e57991a058efb85d787a545a8c2581
layer 19 sha256:c1e8a3847be63c78e85e860f4d7645688177055f671a40491de2b428ff803873
layer 20 sha256:2cefec28614ea4d58b8fc1498d7e582ca5ce36914946299ef94ef878acae8289
layer 23 sha256:b68227d29e224b2d22a5a20c5224e7ba3803ea0eed1be57ebb5a93de8c3b79dd
layer 24 sha256:b6bb17cb97e768c85b26299b5b0cf2db046282153023a7c64530596a58c1c7f2
mkdir -p "$W/b" && cd "$W/b" && cp "$HERE"/*.cu .
build() {  # build <nvcc> <log> <args...>
  nv=$1; log=$2; shift 2
  if ! "$nv" "$@" > "$log" 2>&1; then
    echo "build failed: $nv $*"; tail -20 "$log"
    echo "(CUDA 13.0's headers need a glibc <= 2.39 host, e.g. Ubuntu 24.04 with g++ 13)"; exit 1
  fi
}
check() {  # check <rebuilt cubin> <layer> <name in the image>
  a=$(sha256sum "$1" | cut -d' ' -f1); b=$(sha256sum "$W/L$2/opt/s1eng/s1eng/$3" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "layer $2 $3: byte-identical ($a)"; else echo "layer $2 $3: DIFFERS (rebuilt $a, image $b)"; fi
}
build "$NVCC134" ngemm9b.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu
build "$NVCC130" ffs.log -arch=sm_120a -O3 -cubin -Xptxas -v -o ffs.cubin ffs.cu
build "$NVCC134" kkt2n_v11.log -cubin -arch=sm_120a -O3 -o kkt2n_v11.cubin kkt2n_v11.cu
build "$NVCC134" ngemm9l.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9l.cubin ngemm9l.cu
build "$NVCC134" ngemm9t.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9t.cubin ngemm9t.cu
build "$NVCC134" kkt2n.log -cubin -arch=sm_120a -O3 -Xptxas -v -o kkt2n.cubin kkt2n.cu
build "$NVCC134" ngr.log -cubin -arch=sm_120a -O3 -Xptxas -v -o ngr.cubin ngr.cu
build "$NVCC134" xa.log -cubin -arch=sm_120a -O3 -Xptxas -v -o xa.cubin xa.cu
check ngemm9b.cubin 16 ngemm9b.cubin
check ffs.cubin 19 ffs.cubin
check kkt2n_v11.cubin 20 kkt2n.cubin
check ngemm9l.cubin 23 ngemm9l.cubin
check ngemm9t.cubin 23 ngemm9t.cubin
check kkt2n.cubin 24 kkt2n.cubin
check ngr.cubin 24 ngr.cubin
check xa.cubin 24 xa.cubin
echo "image: $REF"
rm -rf "$W"
