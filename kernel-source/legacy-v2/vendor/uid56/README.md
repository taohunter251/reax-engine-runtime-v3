# CUDA kernel sources of the REAX s1-fast engine committed on SN92 UID 56

This repository publishes the source of the two compiled kernels that ship without source in the engine image of
subnet 92, UID 56:

- hotkey `5HC3sybJYnAUcEcWkKZnaT6f4fNXFQ5Q42EcZXpEKb8UN7Wr`
- commitment `reax-eng/1:s1-fast:ghcr.io/masterdev777/s1k@sha256:23dcdecc1519edf813699ab48adadd1df07a269016da3816bee37cba84e8a89d`

| file in the image (`/opt/s1eng/s1eng/`) | layer | sha256 of the cubin | source here | sha256 of the source |
|---|---|---|---|---|
| `ngemm9b.cubin` | 16 | `f097e26cfa1064f3e784ce0bb6bf79263491761933f4cd9d9f1ec9ef67545937` | `ngemm9b.cu` | `fc4e85d600ed8caefcbcf753ff94072f9ddb2dc1aeee44617dfb668dd617be55` |
| `ffs.cubin` | 19 | `b48ef79609243e48be2be1d11fa104d6f1725071c3a8879c9b79eec3ae362cc9` | `ffs.cu` | `724abb42dcca68d9f4f2c565dbb7ea7d530c02f6c9a8fd82d476b2762d761b15` |

The image's other compiled kernels (`ngemm5.cubin`, `ngemm5_sm89.cubin`, `ngemm5_sm90.cubin`) are built from
`ngemm5.cu`, which the image already ships next to them in layer 15. Every other engine file in the image is Python
source, and layers 0-14 are the official `ghcr.io/reaxlabs/reax-s1-fast-runtime:1.2.0-rc1` layers.

## Build steps

Each command rebuilds the shipped cubin byte for byte (same sha256):

```
# CUDA 13.4 (nvcc V13.4.92)
nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu

# CUDA 13.0 (nvcc V13.0.88)
nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o ffs.cubin ffs.cu
```

- `-Xptxas -v` matters: ptxas records its own command line in the cubin's `.note.nv.tkinfo` section, so omitting the
  flag changes that note (not the machine code) and therefore the sha256.
- CUDA 13.0.88 is available as NVIDIA's `nvidia-cuda-nvcc` 13.0.88 Python wheel. Its headers need a host with glibc 2.39
  or older; we built it in an Ubuntu 24.04 container with g++ 13.3.
- The kernels target the RTX PRO 6000 Blackwell (`sm_120a`).

## Check against the committed image

`verify.sh` extracts the two cubins from the committed image with `crane`, rebuilds them from the sources here and
compares the sha256:

```
./verify.sh <path to nvcc 13.4> <path to nvcc 13.0>
```

Expected output:

```
ngemm9b.cubin: byte-identical (f097e26cfa1064f3e784ce0bb6bf79263491761933f4cd9d9f1ec9ef67545937)
ffs.cubin: byte-identical (b48ef79609243e48be2be1d11fa104d6f1725071c3a8879c9b79eec3ae362cc9)
```

## Licence

See `LICENSE`.
