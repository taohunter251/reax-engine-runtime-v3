# REAX engine v3

Private frozen release of the combined_nomix candidate. The current runtime
uses the reviewed UID240 image and the retained private projection, gate/up,
local-fold, activation-table and canonical-suffix repair implementations.
Different length buckets remain separate when scheduling private fast paths.

Build with `docker build -t reax-engine:v3 .`. Kernel artifacts must first be
reproduced using the matching private reax-engine-kernels-v3 source repository:
`python ../reax-engine-kernels-v3/reproduce.py --private-only --nvcc130 PATH
--nvcc134 PATH --runtime-directory . --output ../v3-kernel-build`.
The three inference parameter files are copied from the immutable v2 image
in the Dockerfile. Model weights are mounted read-only at runtime.

The image serves the original Unix-socket API using RUNTIME_UDS and
S1M_INTERNAL_TOKEN_FILE. It initializes MPF before CUDA and writes caches only
under the configured writable cache directory. No research dispatcher, saved
requests, benchmark results, wallet material or credentials are included.

## Validation limits

The research candidate passed both measured stock-evaluator sessions, with
99.9023% timed top-answer agreement and zero repeat drift. Six retained hard
precision cases remain unresolved. Those cases are not distributed with the
image. Native measurements do not establish Docker/gVisor qualification or
an official score. Freezing and committing this release does not resolve the
known numerical limitations.

## Attribution

See NOTICE, parents.json and the source credits beside inherited implementations.
The v3 source repositories and registry package remain private until explicitly
authorized for public release. Previous public v2 repositories are unchanged.
