from contextlib import contextmanager
from .core import Checkpoint

class Router:
    def __init__(self, full, families):
        self.core, self.families = full.core, set(families)
        self.policy = full.assets.policy

    def project(self, weight, x, length):
        key = self.policy.by_pointer.get(weight.data_ptr())
        if key is None or key[1] not in self.families:
            return None
        return self.policy.project(*key, x, length)

    @contextmanager
    def context(self):
        core, policy = self.core, self.policy
        previous = core.gemm, core.pair, core.lin_res, core._gdn
        gemm, pair, lin_res, gdn = previous
        def matrix(weight, x, length):
            value = self.project(weight, x, length)
            return gemm(weight, x, length) if value is None else value
        def residual_projection(module, x, length):
            value = self.project(module.weight, x, length)
            return lin_res(module, x, length) if value is None else value
        def paired(layer, name, left, right, x, length):
            index = policy.indices[id(layer)]
            if name == 'qkv_z' and 'gdn_input' in self.families and (index, 'gdn_input') in policy.weights:
                value = policy.project(index, 'gdn_input', x, length)
                if value is not None:
                    return value[:, :len(left.weight)], value[:, len(left.weight):]
            return pair(layer, name, left, right, x, length)
        def gated(layer, x, meta, index):
            enabled = ('gdn_input' in self.families and (index, 'gdn_input') in policy.weights
                       and policy.source['chunk_for'].get((index, 'gdn_input', meta.L)) is not None)
            if not enabled:
                return gdn(layer, x, meta, index)
            parallel = core.par
            core.par = parallel-{'z'}
            try:
                return gdn(layer, x, meta, index)
            finally:
                core.par = parallel
        core.gemm, core.pair, core.lin_res, core._gdn = matrix, paired, residual_projection, gated
        try:
            yield
        finally:
            core.gemm, core.pair, core.lin_res, core._gdn = previous


class MergedCheckpoint(Checkpoint):
    def __init__(self, full, families, first=12, start=12):
        super().__init__(full, first, start)
        self.router = Router(full, families)
        self.gemm_marks = (*full.gemm_marks, 'slice_', 'hybrid8_')

    def _suffix(self, meta, h, x, approximate=False):
        if approximate:
            with self.router.context():
                return super()._suffix(meta, h, x, approximate)
        return super()._suffix(meta, h, x, approximate)

