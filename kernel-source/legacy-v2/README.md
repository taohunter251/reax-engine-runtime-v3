# reax-engine kernel sources

Source and reproducible build instructions for the frozen reax-engine v2 CUDA
binaries, including all six inherited components requested in the UID57 review.
This repository contains no runtime code, model parameters or post-v2 research.
Frozen v2 runtime source is at https://github.com/taohunter251/reax-engine-runtime.
Owner award approval is not claimed.

| Component | Complete build input | Frozen binary SHA-256 |
| --- | --- | --- |
| Mixed INT8/BF16 gate/up projection and SwiGLU | `src/native_hybrid8.cu`, `src/hybrid8_tiles.cuh` | `59af3954d443b50df512f58d9156519f0945425cc4b73266d3c43e2697b8ac02` |
| Sliced mixed-precision projections | `src/sliced_linear8.cu`, `src/hybrid8_tiles.cuh` | `664c30f9fdbf0a94006b9706b8d60f38ff0867cfec61604221eb9a387136504b` |
| Block rotation and activation quantization | `src/block_rotation.cu` | `7857f2063a8097fff50391067ef38de99ec2b104aba8d2ecf7d80cd43f9375df` |

`manifest.json` records the source/header hashes and the exact paths of these
binaries in the prepared image. A single cubin contains multiple launch variants.
`src/activation_quantize.py` supplies the additional readable Triton activation
preparation and slice-reduction kernels; Triton compiles those during startup.

## Rebuild

Use Linux x86-64, Python 3.11 or newer, CUDA nvcc **V13.0.88**, and its compatible
host C++ compiler. The target is Blackwell **sm_120a**. The frozen binaries were
compiled with `-O3 -arch=sm_120a --cubin -Xptxas=-v`.

```sh
python3 build.py --nvcc /usr/local/cuda-13.0/bin/nvcc
```

The script checks every input hash, builds all three objects and requires the
complete output file SHA-256 to match the table. It does not strip ELF sections
or normalize binaries before comparison. No GPU, model weights, calibration
requests or inference answers are needed for compilation.

To compare directly against files extracted from the image as well:

```sh
python3 build.py --shipped /path/to/extracted/opt/reax-v2/parameters/kernels
```

`build/verification.json` contains the result. The original compiler commands and
input hashes are retained under `reproducibility/`; their absolute paths describe
the original build environment and are not required by the portable build.

All three objects were independently rebuilt from this portable source layout on
2026-10-22 using nvcc V13.0.88. Every complete file matched the corresponding
frozen runtime object byte for byte. The retained results are in
`reproducibility/verification.json`.

## Reused sources and attribution

The first two kernels are derivatives of the NG5 matrix tile, load pipeline and
epilogue helpers. They are custom modifications, not independently authored
reimplementations of those helpers. `vendor/ng5/ngemm5.cu` preserves the upstream
source obtained from the declared UID56 base image; its exact SHA-256 and parent
digest are in `manifest.json`. The complete generated header used to build the
custom binaries is included verbatim under `src/`.

The runtime's unchanged FFS and NGEMM9B objects have published MIT source in
[masterdev777/reax-uid56-kernels](https://github.com/masterdev777/reax-uid56-kernels/tree/93524e7d00252b96f0619f0c76e12f7da4f50c1e).
That source, its original license and build instructions are preserved under
`vendor/uid56/`. Their compiler versions differ: NGEMM9B requires nvcc 13.4.92,
whereas FFS requires nvcc 13.0.88. The three custom objects above use 13.0.88.

Our additions are MIT licensed; retain the upstream license and notices with
reused source. The source repository and compiler verification do not replace
the required `ai.reax.engine.parents` label or the owner's award review.

## Inherited kernels disclosed for the UID57 correction

All six complete binaries rebuilt byte-for-byte with CUDA **V13.4.92**:

| Kernel | Source | Build flags in addition to `-cubin -arch=sm_120a -O3 -Xptxas -v` |
| --- | --- | --- |
| kkt2n | `vendor/uid176/kkt2n.cu` | none |
| ngemm9l | `vendor/uid176/ngemm9l.cu` | none |
| ngemm9t | `vendor/uid176/ngemm9t.cu` | none |
| ngr | `vendor/uid176/ngr.cu` | none |
| xa | `vendor/uid176/xa.cu` | none |
| pron | `vendor/uid164/pron.cu` | `-DPRON_SHIP` |

```sh
python3 build_inherited.py --nvcc /path/to/CUDA-13.4.92/bin/nvcc
# Optionally compare with the actual extracted image objects:
python3 build_inherited.py --nvcc /path/to/CUDA-13.4.92/bin/nvcc --shipped /path/to/opt/reax-v2/vendor/s1eng
```

`inherited-manifest.json` pins every source and binary hash. Full-file rebuild
results are in `reproducibility/inherited-verification.json`. No ELF sections
were removed or normalized. PRON_SHIP selects the one shipped `pron4_r4` entry;
without it the source builds extra development variants and the file differs.

Five sources and their MIT license are preserved from
[masterdev777/reax-uid176-kernels](https://github.com/masterdev777/reax-uid176-kernels/tree/b66f95111d196f4994a79ca786572318d40aeeb5).
The PRON source origin, intermediate delivery digest, original author and MIT
image license declaration are recorded in `vendor/uid164/NOTICE`. The scoring
parent label names UID56, UID176 and UID164 as requested by the subnet owner's
updated review. UID164's reviewed parent digest is
`944fa7aee414607c4b620de5600224fb7b844f0f9d844031b07f591b8b7455f7`;
its PRON implementation and original authorship remain explicitly attributed.

CUDA source for the inherited NG5 architecture variants is retained at
`vendor/ng5/ngemm5.cu`; the pinned base image also ships it next to its binaries.
All original upstream comments, attribution and licenses are retained.
Frozen v2 runtime source is published separately in reax-engine-runtime.
Publication of sources does not constitute owner award approval.
