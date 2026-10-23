"""Keep the peer's current execution class and plans in a private checkpoint adapter."""
import copy
from contextlib import contextmanager

import torch
from reaxv2.projection import MergedCheckpoint


class NativeCheckpoint(MergedCheckpoint):
    """Retain UID240's full forward, recording the canonical repair boundary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from s1eng import skr
        self.skr_module, self.prune_counts = skr, []

    def __call__(self, meta, approximate=False):
        from contextlib import nullcontext

        core = self.full.core
        old_mlp, old_add = core._mlp, core.add_norm
        boundary = core.ff.layers[self.start]['ln1_w'].data_ptr()
        state = []

        def mlp(layer, x, length, defer=False):
            index = self.gup.retained['indices'][id(layer)]
            if approximate and index >= self.gup.first and index in self.gup.bank:
                return self.gup.mlp(layer, x, length, True, defer)
            return old_mlp(layer, x, length, defer=defer)

        def add(h, y, weight, eps):
            result = old_add(h, y, weight, eps)
            if weight.data_ptr() == boundary:
                state[:] = result
            return result

        with self.full.workspace(), (self.router.context() if approximate else nullcontext()):
            core._mlp, core.add_norm = mlp, add
            try:
                value = core(meta)
                assert len(state) == 2, 'Canonical checkpoint was not recorded'
                return value, tuple(state)
            finally:
                core._mlp, core.add_norm = old_mlp, old_add

    def suffix(self, meta, h, x, approximate=False):
        core = self.full.core
        old = core._mx, core._mx_last
        core._mx = (core.mxp[(meta.L, meta.S)], meta) if getattr(meta, 'S', 0) else None
        core._mx_last = False
        try:
            return super().suffix(meta, h, x, approximate)
        finally:
            core._mx, core._mx_last = old

    def _suffix(self, meta, h, x, approximate=False):
        # Set the mixed last-row boundary at the exact call site used by the
        # checkpoint suffix implementation; ordinary packed graphs are unchanged.
        core = self.full.core
        old = core._attn_last

        def last(*args, **kwargs):
            core._mx_last = core._mx is not None
            return old(*args, **kwargs)

        core._attn_last = last
        try:
            return super()._suffix(meta, h, x, approximate)
        finally:
            core._attn_last = old


class NativeFull:
    def __init__(self, gpu, assets):
        import s1eng
        from s1eng import kprio

        self.package = s1eng
        self.original = gpu.m.pf
        self.core = copy.copy(self.original)
        self.core.bf = copy.copy(self.original.bf)
        self.core.pinned = copy.copy(self.original.pinned)
        for key, value in vars(self.original.pinned).items():
            if isinstance(value, dict):
                setattr(self.core.pinned, key, dict(value))
        self.core.bf.pinned = self.core.pinned
        for key in ('_ar', '_sides', 'rarn'):
            value = getattr(self.original, key, None)
            if isinstance(value, dict):
                setattr(self.core, key, dict(value))
        self.core._sides = {}
        self.assets = assets
        self.gemm_marks = kprio.GEMM_MARKS
        self.workspaces = {}
        self.side_workspaces = {}
        self.plan = dict(
            source='UID240 native PackedForward, unchanged saved plans',
            par=sorted(self.core.par),
            native_class=type(self.core).__module__ + '.' + type(self.core).__name__,
        )

    @contextmanager
    def workspace(self):
        stream = torch.cuda.current_stream().cuda_stream
        if stream not in self.workspaces:
            size = max(self.original.bf.ws.numel(), self.core.pinned.ws_bytes, 1 << 20)
            self.workspaces[stream] = torch.empty(size, device='cuda', dtype=torch.uint8)
            self.side_workspaces[stream] = torch.empty(size, device='cuda', dtype=torch.uint8)
        previous = self.core.bf.ws, self.core.ws_side
        self.core.bf.ws = self.workspaces[stream]
        self.core.ws_side = self.side_workspaces[stream]
        try:
            yield self.core
        finally:
            self.core.bf.ws, self.core.ws_side = previous

    def __call__(self, meta):
        with self.workspace() as core:
            return core(meta)
