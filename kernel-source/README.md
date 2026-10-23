# Private REAX engine v3 kernel sources

This repository freezes the CUDA/C sources and build recipes for v3. It includes
the inherited UID240 sources and their complete original KERNELS.md, the v2
source disclosure, and the selected private post-v2 kernel implementations.
Original licenses and author credits are preserved; see NOTICE and parents.json.

`manifest-v3.json` records source checksums, shipped binary checksums, compiler
versions, flags and runtime paths. Run `python reproduce.py --private-only
--nvcc130 PATH --nvcc134 PATH --output build --runtime-directory RUNTIME_CHECKOUT`
to rebuild, verify and install the v3 additions. All eight private additions were
rebuilt byte-identically; see reproducibility/v3-private-verification.json.
`legacy-v2/build.py` rebuilds the unchanged v2 parameter-bank kernels.
The UID240 objects retain their exact published build recipes and source hashes
in vendor/uid240/KERNELS.md and manifest-v3.json; they remain in the pinned base.
The original source path is retained in the manifest because some CUDA module
identifiers depend on it. For those kernels, reproduce at the documented original
path inside an isolated build container; never replace a live installation.

No weights, wallet material, credentials, saved evaluation inputs or answers are
included. This repository and the corresponding v3 runtime and image stay private.
