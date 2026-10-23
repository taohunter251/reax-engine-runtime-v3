"""Native PRO 6000 development tests for per-token FP8 MLP projections.

Approximate products are used only in this disposable experiment. The unchanged
champion provides control logits; no release code or model weights are modified.
"""
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'precision-output'
QUANT_IMPL = 'torch'


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + '\n')


def emit(stage, **values):
    print(json.dumps({'stage': stage, **values}), flush=True)


def fp8_product(x, weight8, weight_scale):
    rows = len(x)
    if rows % 16:
        x = torch.nn.functional.pad(x, (0, 0, 0, (-rows) % 16))
    if QUANT_IMPL == 'fused':
        from fp8_quant import quantize
        a8, scale = quantize(x)
    else:
        scale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1.e-12) / 448.
        a8 = (x.float() / scale).to(torch.float8_e4m3fn)
    result = torch._scaled_mm(a8, weight8.T, scale_a=scale, scale_b=weight_scale.T,
                               out_dtype=torch.bfloat16, use_fast_accum=False)
    return result[:rows]


def graph_time(call, repeats=24):
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats, result


def make_entries(encoder, gpu, documents, count=16):
    from s1eng.model import Job, bucket_of
    from s1eng.prompt import options_of, prompt_text
    groups = {}
    for document in documents:
        question = next(iter(document['questions'].values()))
        ids = encoder.encode(prompt_text(document['state'], question.get('instructions', ''),
                                        [v for _, v in options_of(question)]))
        groups.setdefault(bucket_of(len(ids)), []).append((document, ids))
    lane = gpu.lanes[0]
    for L, seqs in groups.items():
        if L not in (128, 192, 256, 320):
            continue
        cap = min(count, max(lane.pbs[L]))
        for offset in range(0, len(seqs), cap):
            part = seqs[offset:offset + cap]
            jobs = []
            for document, ids in part:
                job = Job(np.asarray(ids, dtype=np.int64), lambda value: None)
                gpu.submit(job)
                jobs.append(job)
            with gpu.cv:
                gpu.q.clear()
            Bs = lane.packed_b(L, len(jobs))
            rows = sum(j.n - j.pl for j in jobs)
            entry = next(item for item in lane.packed[L, Bs] if item[0] >= rows)
            slot = gpu.free[0]
            gpu._launch(lane, L, jobs, slot, (Bs, entry))
            slot.event.synchronize()
            yield part, entry


@torch.inference_mode()
def main(encoder, gpu):
    global QUANT_IMPL
    import reax_subnet.engrunner as er
    from reax_subnet.engcomp import check_correctness, max_distance
    from reax_subnet.engvalidator import TRACKS
    from s1eng.server import Engine
    from s1eng.prompt import options_of
    params = TRACKS['s1-fast']
    OUTPUT.mkdir(exist_ok=True)
    requests = {name: [d.body() for d in er.fresh_decisions(n, params.pool)]
                for name, n in [('calibration', 256), ('heldout', 512)]}
    write('requests.json', requests)
    f, lane = gpu.m.pf, gpu.lanes[0]
    old_gemm, old_ws = f.gemm, f.bf.ws
    f.bf.ws = lane.ws[0]
    selected = {}
    for i in (3, 15, 30):
        weight = f.ff.layers[i]['mlp'].down_proj.weight
        scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1.e-12) / 448.
        w8 = (weight.float() / scale).to(torch.float8_e4m3fn)
        selected[weight.data_ptr()] = {'layer': i, 'weight': weight, 'w8': w8, 'scale': scale}
    samples = {}

    def observe(weight, x, L):
        if weight.data_ptr() in selected and weight.data_ptr() not in samples:
            samples[weight.data_ptr()] = (x.clone(), L)
        return old_gemm(weight, x, L)

    f.gemm = observe
    first_part, first_entry = next(make_entries(encoder, gpu, requests['calibration']))
    baseline_logits = f(first_entry[2]).clone()
    torch.cuda.synchronize()
    f.gemm = old_gemm
    rows = []
    try:
        quant_checks = []
        for pointer, cfg in selected.items():
            x, L = samples[pointer]
            wanted = old_gemm(cfg['weight'], x, L)
            try:
                from fp8_quant import quantize
                scale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1.e-12) / 448.
                a8 = (x.float() / scale).to(torch.float8_e4m3fn)
                fused_a8, fused_scale = quantize(x)
                quant_exact = torch.equal(a8.view(torch.uint8), fused_a8.view(torch.uint8)) and torch.equal(scale, fused_scale)
                quant_checks.append({'layer':cfg['layer'],'quantized_bytes_and_scales_exact':quant_exact})
                QUANT_IMPL = 'fused' if quant_exact else 'torch'
                trial = fp8_product(x, cfg['w8'], cfg['scale'])
                torch.cuda.synchronize()
                smaller = fp8_product(x[:16].contiguous(), cfg['w8'], cfg['scale'])
                batch_shape_exact = torch.equal(smaller.view(torch.int16), trial[:16].view(torch.int16))
                timings = []
                for _ in range(3):
                    for arm in ('base', 'fp8', 'fp8', 'base'):
                        call = (lambda: old_gemm(cfg['weight'], x, L)) if arm == 'base' else \
                               (lambda: fp8_product(x, cfg['w8'], cfg['scale']))
                        elapsed, _ = graph_time(call)
                        timings.append({'arm': arm, 'ms': elapsed})
                means = {arm: statistics.median(r['ms'] for r in timings if r['arm'] == arm) for arm in ('base', 'fp8')}
                row = {'layer': cfg['layer'], 'projection': 'down', 'rows': len(x), 'K': x.shape[1],
                       'N': len(cfg['weight']), 'rowwise_scaled_mm_supported': True,
                       'activation_quantization': QUANT_IMPL,
                       'first_16_rows_exact_at_different_batch_size': batch_shape_exact,
                       'bf16_exact_cell_fraction': float((trial.view(torch.int16) == wanted.view(torch.int16)).float().mean()),
                       'milliseconds': means, 'speedup': means['base'] / means['fp8'], 'timings': timings}
            except Exception:
                row = {'layer': cfg['layer'], 'rowwise_scaled_mm_supported': False, 'error': traceback.format_exc()}
            rows.append(row)
            write('matrix-tests.json', rows)
            write('quantizer-tests.json', quant_checks)
            emit('matrix_test', **{k:v for k,v in row.items() if k != 'timings'})
        if not all(r.get('rowwise_scaled_mm_supported') for r in rows):
            write('summary.json', {'scope': 'GPU development test only; unsupported route; no release changes', 'matrix_tests': rows})
            return
        # One shared deterministic quantizer implementation for every projection probe.
        QUANT_IMPL = 'fused' if all(r['quantized_bytes_and_scales_exact'] for r in quant_checks) else 'torch'

        engine = Engine.__new__(Engine)
        engine.temperature = 2.07
        all_results = []
        # Probe one layer at a time. Candidate selection must not use held-out answers.
        for name, documents in requests.items():
            outputs = {str(cfg['layer']): {'reference': {}, 'candidate': {}} for cfg in selected.values()}
            batches = 0
            for part, entry in make_entries(encoder, gpu, documents):
                meta = entry[2]
                wanted = f(meta).clone().float().cpu().numpy()
                for pointer, cfg in selected.items():
                    def changed(weight, x, L, p=pointer, config=cfg):
                        return fp8_product(x, config['w8'], config['scale']) if weight.data_ptr() == p else old_gemm(weight, x, L)
                    f.gemm = changed
                    try:
                        actual = f(meta).clone().float().cpu().numpy()
                    finally:
                        f.gemm = old_gemm
                    target = outputs[str(cfg['layer'])]
                    for b, (document, ids) in enumerate(part):
                        question = next(iter(document['questions'].values()))
                        options = options_of(question)
                        reference = engine._answer(question['type'], options, wanted[b])
                        candidate = engine._answer(question['type'], options, actual[b])
                        target['reference'][document['id']] = list(reference['probabilities'].values())
                        target['candidate'][document['id']] = list(candidate['probabilities'].values())
                batches += 1
            for layer, outputs_one in outputs.items():
                result = check_correctness(outputs_one['candidate'], outputs_one['reference'], params)
                mismatches = [ident for ident in outputs_one['reference'] if
                    np.argmax(outputs_one['reference'][ident]) != np.argmax(outputs_one['candidate'][ident])]
                row = {'dataset': name, 'layer': int(layer), 'items': len(outputs_one['reference']),
                       'batch_count': batches, 'ok': result.ok, 'reason': result.reason,
                       'mean_distance': result.mean_distance, 'max_distance': result.max_distance,
                       'top_agreement': result.top_agreement, 'top_mismatches': mismatches}
                all_results.append(row)
                write(f'answers-{name}-layer-{layer}.json', outputs_one)
                emit('output_test', **row)
            write('output-tests.json', all_results)
        serial_outputs = {str(cfg['layer']): {} for cfg in selected.values()}
        for part, entry in make_entries(encoder, gpu, requests['heldout'][:24], count=1):
            document = part[0][0]
            question = next(iter(document['questions'].values()))
            options = options_of(question)
            for pointer, cfg in selected.items():
                def changed(weight, x, L, p=pointer, config=cfg):
                    return fp8_product(x, config['w8'], config['scale']) if weight.data_ptr() == p else old_gemm(weight, x, L)
                f.gemm = changed
                try:
                    logits = f(entry[2]).float().cpu().numpy()[0]
                finally:
                    f.gemm = old_gemm
                answer = engine._answer(question['type'], options, logits)
                serial_outputs[str(cfg['layer'])][document['id']] = list(answer['probabilities'].values())
        serial_results = []
        for layer, answers in serial_outputs.items():
            reference = {ident: outputs[layer]['candidate'][ident] for ident in answers}
            distance = max_distance(answers, reference, params.temperature)
            serial_results.append({'layer': int(layer), 'items': len(answers), 'max_distance': distance,
                                   'within_determinism_threshold': distance <= params.determinism_tau})
        write('serial-comparisons.json', {'results': serial_results, 'answers': serial_outputs})
        write('summary.json', {'scope': 'Native GPU development tests; gate metrics against champion, not official reference qualification',
            'matrix_tests': rows, 'output_tests': all_results, 'release_changed': False,
            'per_token_scaling': True, 'conditional_batch_precision': False,
            'serial_checks': serial_results, 'activation_quantization':QUANT_IMPL})
    finally:
        f.gemm, f.bf.ws = old_gemm, old_ws


if __name__ == '__main__':
    import s1eng.server as server
    def initialize(self, encoder, gpu, *args, **kwargs):
        try:
            with torch.inference_mode():
                main(encoder, gpu)
        except BaseException:
            OUTPUT.mkdir(exist_ok=True)
            (OUTPUT/'exception.txt').write_text(traceback.format_exc())
            raise
        raise SystemExit(0)
    server.Engine.__init__ = initialize
    from s1eng.main import main as start
    start()
