"""Our own forward for the Qwen3.5 text model (batch 1, right-padded bucket length), op for op equal to the
reference's captured forward, with fused kernels switched in per op family.

impl string: "own" = pure torch replica (the same aten ops, shapes and strides as transformers' forward under graph
capture, so the same kernels run); "own+<f1>+<f2>..." enables fused families from fused.py:
    rms   RMSNorm (+ the residual add that precedes it)  [decoder norms, final norm]
    ...   (see FAMILIES below)
Constant per-length work (rotary tables, the causal mask, 1+w of the zero-centred norms) is computed once per length
at capture time instead of in every replay; values are identical to what the reference computes in the graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FAMILIES = ("rms", "gnorm", "qkn", "conv", "glue", "swiglu", "attnglue", "fmask", "skip")


class FastForward:
    def __init__(self, hf_model, slot_weight: torch.Tensor, impl: str, fused_w: dict | None = None):
        self.hf = hf_model
        self.tm = hf_model.model
        self.cfg = self.tm.config
        self.slot_weight = slot_weight
        parts = impl.split("+")
        assert parts[0] == "own", impl
        self.on = set(parts[1:])
        bad = self.on - set(FAMILIES)
        if bad:
            raise SystemExit(f"unknown fused families {bad}")
        if self.on:
            from . import fused  # noqa: F401  (Triton kernels)
            self.fused = fused
        self.eps = self.cfg.rms_norm_eps
        self.layers = []
        fused_w = fused_w or {}
        for li, layer in enumerate(self.tm.layers):
            d = {"type": layer.block_type, "ln1": layer.input_layernorm, "ln2": layer.post_attention_layernorm,
                 "mlp": layer.mlp, "fused": {g: blk for (i, g), blk in fused_w.items() if i == li}}
            # 1 + w (float32) of the zero-centred norms: the reference recomputes it in every forward
            d["ln1_w"] = (1.0 + layer.input_layernorm.weight.float())
            d["ln2_w"] = (1.0 + layer.post_attention_layernorm.weight.float())
            if layer.block_type == "linear_attention":
                la = layer.linear_attn
                d["la"] = la
                d["conv_w"] = la.conv1d.weight.squeeze(1)              # [C, 4] (view, as the reference passes it)
                d["conv_w3"] = d["conv_w"].unsqueeze(1)                 # [C, 1, 4]
                d["negexpA"] = -la.A_log.float().exp()                  # [32] f32
            else:
                sa = layer.self_attn
                d["sa"] = sa
                d["qn_w"] = (1.0 + sa.q_norm.weight.float())
                d["kn_w"] = (1.0 + sa.k_norm.weight.float())
            self.layers.append(d)
        self.fn_w = (1.0 + self.tm.norm.weight.float())
        self.consts: dict[int, tuple] = {}
        self.rec = None  # tests: list that collects (family, layer, tensors) at fused-op boundaries (torch path)

    def _r(self, fam, i, *ts):
        if self.rec is not None:
            self.rec.append((fam, i, tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in ts)))

    # ------------------------------------------------------------------ per-length constants
    def _consts(self, ids: torch.Tensor):
        n = ids.shape[1]
        c = self.consts.get(n)
        if c is None:
            from transformers.masking_utils import create_causal_mask

            with torch.inference_mode():
                emb = self.tm.embed_tokens(ids)
                position_ids = torch.arange(n, device=ids.device).view(1, 1, -1).expand(4, 1, -1)
                text_pos, pos = position_ids[0], position_ids[1:]
                mask = create_causal_mask(config=self.cfg, inputs_embeds=emb, attention_mask=None,
                                          past_key_values=None, position_ids=text_pos, allow_is_causal_skip=False)
                cos, sin = self.tm.rotary_emb(emb, pos)
                if "fmask" in self.on:
                    # the additive mask SDPA derives from a bool mask (convert_boolean_attn_mask), computed once
                    mask = torch.where(mask, 0.0, torch.scalar_tensor(float("-inf"), dtype=emb.dtype, device=emb.device))
            c = (mask, cos, sin)
            self.consts[n] = c
        return c

    # ------------------------------------------------------------------ reference-equal torch ops
    def _rms_torch(self, x, w1p):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * w1p).type_as(x)

    def _gated_norm_torch(self, la, o, z):
        # Qwen3_5RMSNormGated.forward
        hs = o.to(torch.float32)
        var = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(var + la.norm.variance_epsilon)
        hs = la.norm.weight * hs.to(torch.bfloat16)
        hs = hs * F.silu(z.to(torch.float32))
        return hs.to(torch.bfloat16)

    def _gdn(self, d, x):
        la = d["la"]
        B, L, _ = x.shape
        fz = self.fused if self.on else None
        mixed_qkv = la.in_proj_qkv(x)
        z = la.in_proj_z(x)
        z = z.reshape(B, L, -1, la.head_v_dim)
        b = la.in_proj_b(x)
        a = la.in_proj_a(x)
        if "conv" in self.on:
            # [B, L, C] in, [B, L, C] out (bf16 conv then SiLU, as the reference), q/k repeat + v made contiguous
            q, k, v = fz.conv_silu_split(mixed_qkv, d["conv_w"], la.key_dim, la.value_dim,
                                         la.num_v_heads // la.num_k_heads, la.head_k_dim, la.head_v_dim)
        else:
            mixed_in = mixed_qkv
            mixed_qkv = mixed_qkv.transpose(1, 2)
            mixed_qkv = F.conv1d(mixed_qkv, d["conv_w3"], None, padding=3, groups=mixed_qkv.shape[1])[:, :, :L]
            mixed_qkv = F.silu(mixed_qkv)
            mixed_qkv = mixed_qkv.transpose(1, 2)
            q, k, v = torch.split(mixed_qkv, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
            q = q.reshape(B, L, -1, la.head_k_dim)
            k = k.reshape(B, L, -1, la.head_k_dim)
            v = v.reshape(B, L, -1, la.head_v_dim)
            rep = la.num_v_heads // la.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
            self._r("conv", self._li, mixed_in, d["conv_w"], q, k, v.contiguous())
        if "glue" in self.on:
            beta, g = fz.gdn_gate(b, a, d["negexpA"], la.dt_bias)
        else:
            beta = b.sigmoid()
            g = d["negexpA"] * F.softplus(a.float() + la.dt_bias)
            self._r("glue", self._li, b, a, d["negexpA"], la.dt_bias, beta, g)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                      use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        o = o.reshape(-1, la.head_v_dim)
        z = z.reshape(-1, la.head_v_dim)
        if "gnorm" in self.on:
            o = fz.gated_rmsnorm(o, z, la.norm.weight, la.norm.variance_epsilon)
        else:
            o_in = o
            o = self._gated_norm_torch(la, o, z)
            self._r("gnorm", self._li, o_in, z, la.norm.weight, o)
        o = o.reshape(B, L, -1)
        return la.out_proj(o)

    def _attn(self, d, x, mask, cos, sin):
        sa = d["sa"]
        B, L, _ = x.shape
        hd = sa.head_dim
        fz = self.fused if self.on else None
        rep = sa.num_key_value_groups
        qg = sa.q_proj(x).view(B, L, -1, hd * 2)
        kr = sa.k_proj(x).view(B, L, -1, hd)
        vr = sa.v_proj(x).view(B, L, -1, hd)
        if "qkn" in self.on:
            q = fz.q_norm_rope(qg, d["qn_w"], cos, sin, self.eps)                         # [B, H, L, hd]
            k, v = fz.kv_norm_rope_rep(kr, vr, d["kn_w"], cos, sin, self.eps, rep)       # [B, H, L, hd] each
        else:
            self._r("qkv_raw", self._li, qg, kr, vr, d["qn_w"], d["kn_w"], cos, sin, rep)
            q, _gate = torch.chunk(qg, 2, dim=-1)
            q = self._rms_torch(q.view(B, L, -1, hd), d["qn_w"]).transpose(1, 2)
            k = self._rms_torch(kr, d["kn_w"]).transpose(1, 2)
            q, k = _rope(q, k, cos, sin)
            v = vr.transpose(1, 2)
            k = _repeat_kv(k, rep)
            v = _repeat_kv(v, rep)
        self._r("sdpa_in", self._li, q, k, v)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=sa.scaling, is_causal=False)
        if "attnglue" in self.on:
            o = fz.attn_gate(o, qg)
        else:
            o_sdpa = o
            gate = torch.chunk(qg, 2, dim=-1)[1].reshape(B, L, -1)
            o = o.transpose(1, 2).contiguous()
            o = o.reshape(B, L, -1).contiguous()
            o = o * torch.sigmoid(gate)
            self._r("attn_gate", self._li, o_sdpa, qg, o)
        return sa.o_proj(o)

    def _mlp(self, d, x):
        mlp = d["mlp"]
        if "swiglu" in self.on:
            return mlp.down_proj(self.fused.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))
        g, u = mlp.gate_proj(x), mlp.up_proj(x)
        y = F.silu(g) * u
        self._r("swiglu", self._li, g, u, y)
        return mlp.down_proj(y)

    # ------------------------------------------------------------------ forward
    def __call__(self, ids: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        mask, cos, sin = self._consts(ids)
        h = self.tm.embed_tokens(ids)
        rms = "rms" in self.on
        fz = self.fused if self.on else None
        pending = None  # residual add folded into the next norm (fused path)
        for i, d in enumerate(self.layers):
            self._li = i
            if rms:
                if pending is None:
                    x = fz.rmsnorm(h, d["ln1_w"], self.eps)
                else:
                    h, x = fz.add_rmsnorm(h, pending, d["ln1_w"], self.eps)
            else:
                x = self._rms_torch(h, d["ln1_w"])
                self._r("rms_in", i, h, d["ln1_w"], x)
            if d["type"] == "linear_attention":
                y = self._gdn(d, x)
            else:
                y = self._attn(d, x, mask, cos, sin)
            if rms:
                h, x = fz.add_rmsnorm(h, y, d["ln2_w"], self.eps)
            else:
                h0 = h
                h = h + y
                x = self._rms_torch(h, d["ln2_w"])
                self._r("rms_add", i, h0, y, d["ln2_w"], h, x)
            y = self._mlp(d, x)
            if rms:
                pending = y
            else:
                h = h + y
        if rms:
            h, hn = fz.add_rmsnorm(h, pending, self.fn_w, self.eps)
        else:
            hn = self._rms_torch(h, self.fn_w)
        return hn[torch.arange(ids.shape[0], device=ids.device), last] @ self.slot_weight.T


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rd = cos.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)
