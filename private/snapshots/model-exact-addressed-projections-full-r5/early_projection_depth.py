"""Measure the model-specific BF16-prefix cost before changing serving.

Fixed, unlabeled activation moments define extra banks for layers 4..11. The
original bucket, prompt fields, model weights and canonical repair remain intact.
This experiment does not choose channels, weights or recipes using answers.
"""
from contextlib import contextmanager
import copy
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from earlier_gate_up_screen import bank as gate_bank
from pipeline_screen import timing
from probe_utils import make_entries
from reaxv2.projection import MergedCheckpoint


def hadamard(value, block):
    shape = value.shape
    value = value.reshape(-1, block)
    for step in (1 << i for i in range(int(math.log2(block)))):
        pair = value.reshape(-1, block // (2 * step), 2, step)
        value = torch.cat((pair[:, :, 0] + pair[:, :, 1],
                           pair[:, :, 0] - pair[:, :, 1]), dim=-1).reshape(-1, block)
    return value.reshape(shape)


def rotated_bank(weight, stats, template):
    n, width = weight.shape
    p = copy.copy(template)
    assert p.width == width and p.hp == 0
    center = (stats['sum'] / stats['count']).float()
    codes, scales, biases = [], [], []
    for part in range(p.parts):
        start, end = part * p.chunk, min(width, (part + 1) * p.chunk)
        w = weight[:, start:end].float()
        bias = (w.double() @ center[start:end].double()).float()
        padded = F.pad(w, (0, p.group - w.shape[1]))
        columns = torch.arange(p.group, device=weight.device) + start
        sign = torch.where(((columns * 1103515245 + 12345) & 65536) != 0, -1., 1.)
        rotated = hadamard(padded * sign, p.block) / p.block
        scale = rotated.abs().amax(1).clamp_min(1.e-12) / 127.
        codes.append((rotated / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        scales.append(scale)
        biases.append(bias)
    p.centers = center.contiguous()
    p.low_weight = torch.cat(codes, dim=1).contiguous()
    p.high_weight = torch.empty((n, 0), device=weight.device, dtype=weight.dtype)
    p.scales = torch.stack(scales).contiguous()
    p.bias = torch.stack(biases).contiguous()
    return p


def sliced_bank(weight, stats):
    p = gate_bank(weight, stats, 128)
    low = p.low_weight.shape[1]
    group = math.ceil(low / 128) * 128
    p.low_columns = F.pad(p.low_columns, (0, group - low), value=-1).contiguous()
    p.low_weight = F.pad(p.low_weight, (0, group - low)).contiguous()
    p.centers = F.pad(p.centers, (0, group - low)).contiguous()
    p.smoothing = F.pad(p.smoothing, (0, group - low), value=1.).contiguous()
    p.scales = p.scales[None, :].contiguous()
    p.bias = p.bias[None, :].contiguous()
    p.group, p.hp, p.parts, p.chunk = group, 128, 1, p.width
    return p


@contextmanager
def calibration_observers(ctx, stats, first=4, end=12):
    core = ctx.full.core
    pointers = {}
    indices = {id(d): i for i, d in enumerate(core.ff.layers)}
    for i in range(first, end):
        d = core.ff.layers[i]
        pointers[d['mlp'].down_proj.weight.data_ptr()] = i, 'down'
        if d['type'] == 'linear_attention':
            pointers[d['la'].out_proj.weight.data_ptr()] = i, 'gdn_output'
    previous = core.lin_res, core._gdn, core._mlp
    real_rows = [0]

    def observe(key, x):
        x = x[:real_rows[0]].float()
        s = stats.setdefault(key, dict(count=0, sum=torch.zeros(x.shape[1], device=x.device, dtype=torch.float64),
                                      square=torch.zeros(x.shape[1], device=x.device, dtype=torch.float64),
                                      maximum=torch.zeros(x.shape[1], device=x.device)))
        s['count'] += len(x)
        s['sum'] += x.double().sum(0)
        s['square'] += x.double().square().sum(0)
        s['maximum'] = torch.maximum(s['maximum'], x.abs().amax(0))

    def residual(module, x, length):
        if (key := pointers.get(module.weight.data_ptr())) is not None:
            observe(key, x)
        return previous[0](module, x, length)

    def gated(layer, x, meta, index):
        if first <= index < end:
            observe((index, 'gdn_input'), x)
        return previous[1](layer, x, meta, index)

    def mlp(layer, x, length, defer=False):
        index = indices[id(layer)]
        if first <= index < end:
            observe((index, 'gate_up'), x)
        return previous[2](layer, x, length, defer=defer)

    core.lin_res, core._gdn, core._mlp = residual, gated, mlp
    try:
        yield real_rows
    finally:
        core.lin_res, core._gdn, core._mlp = previous


def dedicated(ctx):
    handle = C.c_void_p()
    assert C.CDLL('libcuda.so.1').cuStreamCreateWithPriority(C.byref(handle), C.c_uint(1), C.c_int(0)) == 0
    stream = torch.cuda.ExternalStream(handle.value, device=0)
    ctx.retained.setdefault('research_external_streams', []).append((handle, stream))
    return stream


def capture(ctx, model, meta, approximate=True):
    stream, side = dedicated(ctx), dedicated(ctx)
    ctx.full.core._sides[stream.cuda_stream] = side
    # Eager controls and graph warmup share the model's scratch workspace.
    # Finish the caller's queued work before the new stream can write it.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(meta, approximate)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        value = model(meta, approximate)[0]
    graph.replay()
    torch.cuda.synchronize()
    return graph, value


def main(ctx, output):
    import reax_subnet.engrunner as runner
    from reax_subnet.engvalidator import TRACKS
    from reax_subnet.engcomp import check_correctness
    from s1eng.server import Engine
    from s1eng.prompt import options_of
    calibration = [d.body() for d in runner.fresh_decisions(128, TRACKS['s1-fast'].pool)]
    (output / 'calibration-requests.json').write_text(json.dumps(calibration))
    stats = {}
    with calibration_observers(ctx, stats) as real_rows:
        for _, entry in make_entries(ctx.encoder, ctx.gpu, calibration, count=8):
            real_rows[0] = int((entry[2].slen - entry[2].plen).sum())
            ctx.model(entry[2], False)
    torch.save({str(key): {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
                for key, s in stats.items()}, output / 'unlabeled-moments.pt')
    expected = {(i, f) for i in range(4, 12) for f in ('gate_up', 'down')}
    expected |= {(i, f) for i in range(4, 12) if ctx.full.core.ff.layers[i]['type'] == 'linear_attention'
                 for f in ('gdn_input', 'gdn_output')}
    assert set(stats) == expected, ('calibration coverage', set(stats), expected)
    assets = copy.copy(ctx.assets)
    assets.payload = dict(ctx.assets.payload)
    assets.payload['chunk_for'] = dict(ctx.assets.payload['chunk_for'])
    assets.projections = dict(ctx.assets.projections)
    policy = copy.copy(ctx.assets.policy)
    policy.project = type(policy).project.__get__(policy, type(policy))
    policy.assets = assets
    policy.source = {'chunk_for': assets.payload['chunk_for']}
    policy.weights, policy.by_pointer = dict(policy.weights), dict(policy.by_pointer)
    gate_banks = dict(ctx.assets.gate_up_banks)
    recipes = {
        'early-down4': (4, ('down',), False),
        'early-GDN4': (4, ('gdn_input', 'gdn_output'), False),
        'early-all8': (8, ('down', 'gdn_input', 'gdn_output'), True),
        'early-all4': (4, ('down', 'gdn_input', 'gdn_output'), True),
    }
    for i in range(4, 12):
        d = ctx.full.core.ff.layers[i]
        gate_banks[i] = gate_bank(d['fused']['gate_up'], stats[i, 'gate_up'], 512)
        for family in ('down', 'gdn_input', 'gdn_output'):
            if family != 'down' and d['type'] != 'linear_attention':
                continue
            weight = d['mlp'].down_proj.weight if family == 'down' else (d['fused']['qkv_z'] if family == 'gdn_input' else d['la'].out_proj.weight)
            policy.weights[i, family] = weight
            policy.by_pointer[weight.data_ptr()] = i, family
            chunks = {L: ctx.assets.payload['chunk_for'].get((20, family, L)) for L in (128, 192, 256, 320)}
            for chunk in set(chunks.values()) - {None}:
                template = ctx.assets.projections[20, family, chunk]
                p = sliced_bank(weight, stats[i, family]) if family == 'gdn_input' else rotated_bank(weight, stats[i, family], template)
                assets.projections[i, family, chunk] = p
            for L, chunk in chunks.items():
                assets.payload['chunk_for'][i, family, L] = chunk
    models = {'v2-fast': ctx.model}
    for name, (first, families, early_gate) in recipes.items():
        m = MergedCheckpoint(ctx.full, ('down', 'gdn_input', 'gdn_output'), first=first if early_gate else 12, start=first)
        p = copy.copy(policy)
        p.project = type(p).project.__get__(p, type(p))
        p.assets = copy.copy(assets)
        p.assets.payload = dict(assets.payload)
        p.assets.payload['chunk_for'] = {k: v for k, v in assets.payload['chunk_for'].items()
                                      if k[0] >= 12 or (k[0] >= first and k[1] in families)}
        p.source = {'chunk_for': p.assets.payload['chunk_for']}
        m.router.policy = p
        m.gup.bank = gate_banks
        models[name] = m
    (output / 'fixed-recipes.json').write_text(json.dumps(recipes, indent=2))
    ctx.retained['early_projection_depth_prepared'] = dict(assets=assets, policy=policy, models=models,
                                                          stats=stats, source=str(output))
    def fields(p):
        return {k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in vars(p).items()}
    parameter_file = output/'early-parameters.pt'
    torch.save(dict(gate_up={i:fields(gate_banks[i]) for i in range(4,12)},
                    projections={key:fields(p) for key,p in assets.projections.items() if key[0]<12}), parameter_file)
    with parameter_file.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    (output/'parameter-manifest.json').write_text(json.dumps(dict(path=parameter_file.name,
                                                                bytes=parameter_file.stat().st_size,
                                                                sha256=digest),indent=2))
    fresh = [d.body() for d in runner.fresh_decisions(1024, TRACKS['s1-fast'].pool)]
    assert not ({d['id'] for d in calibration} & {d['id'] for d in fresh})
    retained = json.loads(Path('/opt/reax-v3/private/output/hessian-INT8-fixed-selection-full-model-r1/requests.json').read_text())
    regressions = retained['regressions']
    (output / 'requests.json').write_text(json.dumps(dict(fresh=fresh, regressions=regressions)))
    # Time full graphs before running the full numerical screen. This is not HTTP throughput.
    costs = []
    for number, (_, entry) in enumerate(make_entries(ctx.encoder, ctx.gpu, fresh[:96], count=5)):
        if number >= 6:
            break
        reference = ctx.model(entry[2], False)[0].clone()
        base, _ = capture(ctx, ctx.model, entry[2])
        for name, m in models.items():
            if name == 'v2-fast':
                continue
            canonical = m(entry[2], False)[0]
            if not torch.equal(canonical, reference):
                torch.save(dict(reference=reference.cpu(), candidate=canonical.cpu(), recipe=name,
                                metadata={k:v.cpu() for k,v in vars(entry[2]).items() if isinstance(v,torch.Tensor)}),
                           output/f'canonical-boundary-failure-{number}-{name}.pt')
                raise AssertionError(('moving checkpoint changed canonical output',name))
            candidate, _ = capture(ctx, m, entry[2])
            samples = dict(base=[], candidate=[])
            for _ in range(3):
                for arm in ('base', 'candidate', 'candidate', 'base'):
                    samples[arm].append(timing(base if arm == 'base' else candidate, repeats=4))
            median = {k: statistics.median(v) for k, v in samples.items()}
            costs.append(dict(recipe=name, L=entry[2].L, B=entry[2].B, rows=entry[0],
                              milliseconds=median, speedup=median['base'] / median['candidate']))
        (output / 'graph-costs.json').write_text(json.dumps(costs, indent=2))
    engine = Engine.__new__(Engine)
    engine.temperature = 2.07

    def probabilities(doc, logits):
        question = next(iter(doc['questions'].values()))
        return list(engine._answer(question['type'], options_of(question), logits)['probabilities'].values())

    summary = {}
    for split, docs in (('fresh', fresh), ('regressions', regressions)):
        refs, raw = {}, {}
        for number, (part, entry) in enumerate(make_entries(ctx.encoder, ctx.gpu, docs, count=8)):
            expected_logits = ctx.model(entry[2], False)[0].float().cpu().numpy()
            for b, (doc, _) in enumerate(part):
                refs[doc['id']] = probabilities(doc, expected_logits[b])
                raw[doc['id']] = {'reference': expected_logits[b].tolist(), 'request': doc}
            for name, m in models.items():
                values = m(entry[2], True)[0].float().cpu().numpy()
                for b, (doc, _) in enumerate(part):
                    raw[doc['id']][name] = values[b].tolist()
            if number % 16 == 0:
                print('EARLY_PROJECTION ' + json.dumps(dict(split=split, items=len(refs))), flush=True)
        results = []
        failures = []
        for name in models:
            answers = {ident: probabilities(record['request'], record[name]) for ident, record in raw.items()}
            for ident, row in answers.items():
                if np.argmax(row) != np.argmax(refs[ident]):
                    failures.append(dict(recipe=name, request=raw[ident]['request'], logits=raw[ident],
                                         reference_probabilities=refs[ident], candidate_probabilities=row))
            for margin in (.125, .25, .375, .5):
                corrected, repairs = {}, 0
                for ident, row in answers.items():
                    scores = np.sort(raw[ident][name][:len(refs[ident])])
                    repair = len(scores) > 1 and scores[-1] - scores[-2] <= margin
                    repairs += int(repair)
                    corrected[ident] = refs[ident] if repair else row
                check = check_correctness(corrected, refs, TRACKS['s1-fast'])
                results.append(dict(recipe=name, margin=margin, items=len(docs), repairs=repairs,
                                    post_repair_top_errors=int(sum(np.argmax(corrected[k]) != np.argmax(refs[k]) for k in refs)),
                                    quality_ok=bool(check.ok), mean_distance=float(check.mean_distance), max_distance=float(check.max_distance)))
        summary[split] = results
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / (split + '-raw.json')).write_text(json.dumps(raw))
        (output / (split + '-failures.json')).write_text(json.dumps(failures))
    ctx.retained['early_projection_depth'] = dict(assets=assets, policy=policy, models=models, stats=stats,
                                                 costs=costs, summary=summary, source=str(output))
    return dict(unlabeled_calibration=128, fresh=1024, regressions=len(regressions), recipes=len(models),
                scope='Canonical numerical screen and full GPU graph costs only. No HTTP benchmark or production change.')
