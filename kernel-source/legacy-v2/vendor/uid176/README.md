# CUDA kernel sources of the REAX s1-fast engine committed on SN92 UID 176

Entry: UID 176, hotkey `5F2PvTKK3BGY7FsXoJ7vxMckAYiMJsC4Yi3S5Pjp6czEDi5h`, competition element
`reax-eng/1:s1-fast:ghcr.io/masterdev777/s1n@sha256:01e6e113da15bbc5f992779bb099a55d99b936ab088b64fb4967ef557f72db6d`.

Every compiled kernel the image ships is in `/opt/s1eng/s1eng/`. Each one rebuilds byte for byte from a source in this folder:

| cubin (layer) | sha256 | source in this folder (sha256) | build line | toolkit |
|---|---|---|---|---|
| `ngemm9b.cubin` (16) | `f097e26cfa1064f3e784ce0bb6bf79263491761933f4cd9d9f1ec9ef67545937` | `ngemm9b.cu` (`fc4e85d600ed8cae…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) |
| `ffs.cubin` (19) | `b48ef79609243e48be2be1d11fa104d6f1725071c3a8879c9b79eec3ae362cc9` | `ffs.cu` (`724abb42dcca68d9…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o ffs.cubin ffs.cu` | CUDA 13.0 (`V13.0.88`) |
| `kkt2n.cubin` (20, an older version under the one in layer 24) | `dc05ffcf6f0a15a29e5609ec120f5df875e5a73aa5fc6e6d2e1c8f85bc6cfac0` | `kkt2n_v11.cu` (`d85ff502650143e1…`) | `nvcc -cubin -arch=sm_120a -O3 -o kkt2n.cubin kkt2n_v11.cu` (no `-Xptxas -v`, as its ptxas note records) | CUDA 13.4 |
| `ngemm9l.cubin` (23) | `f35d8d2a2d44d01a0de8154adeaac94651035aaf83aaf04e021f048d37b57e88` | `ngemm9l.cu` (`27a5d2f0eed18ec7…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9l.cubin ngemm9l.cu` | CUDA 13.4 |
| `ngemm9t.cubin` (23) | `111b70b04ac63f4c12f82145d45a1ae64bafb3c5f812fd3dae4b195b08e7f70d` | `ngemm9t.cu` (`a2f47b5a440b1f99…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9t.cubin ngemm9t.cu` | CUDA 13.4 |
| `kkt2n.cubin` (24) | `fd5b50c85669d3f106d3415523b62651bf5e6ace69cb0531404059ba87000c3a` | `kkt2n.cu` (`05f7c62710f2e8ba…`) | `nvcc -cubin -arch=sm_120a -O3 -Xptxas -v -o kkt2n.cubin kkt2n.cu` | CUDA 13.4 |
| `ngr.cubin` (24) | `15dc8936691b884e1364a5c57f64470e354bc59c6c64c1244307fcf614f8e889` | `ngr.cu` (`fe6934567dcea3a7…`) | `nvcc -cubin -arch=sm_120a -O3 -Xptxas -v -o ngr.cubin ngr.cu` | CUDA 13.4 |
| `xa.cubin` (24) | `8d119f6b271de9ccf86d58b99b5a37299cc9460b1c247f4a74e70bd8154762aa` | `xa.cu` (`231ea2f7bfd427b1…`) | `nvcc -cubin -arch=sm_120a -O3 -Xptxas -v -o xa.cubin xa.cu` | CUDA 13.4 |
| `ngemm5.cubin`, `ngemm5_sm89.cubin`, `ngemm5_sm90.cubin` (15) | (see the image) | `ngemm5.cu`, already shipped in the image next to them | `nvcc -arch=sm_120a -O3 -cubin`; sm_89 / sm_90: `-arch=sm_89` / `-arch=sm_90` `-O3 -cubin -Xptxas -v` | CUDA 13.4 / CUDA 13.0 |

Every other engine file in the image is Python source. Layers 0-14 are the official `ghcr.io/reaxlabs/reax-s1-fast-runtime:1.2.0-rc1`
layers.

## How to check

`verify.sh` unpacks the layers above from the committed image (`crane`), rebuilds every cubin from the sources here and compares
the sha256:

```
./verify.sh <path to nvcc 13.4> <path to nvcc 13.0>
```

`-Xptxas -v` matters: ptxas writes its own command line into the cubin's `.note.nv.tkinfo` section. The CUDA 13.0.88 compiler is
NVIDIA's `nvidia-cuda-nvcc` 13.0.88 Python wheel; its headers need a host with glibc 2.39 or older (we used Ubuntu 24.04 with
g++ 13.3).

## Licence

See `LICENSE` (MIT).
