"""Test independent K slices for this model's N64 control projections."""
import json, statistics
import torch
from exact_extension_full_screen import dedicated
from mixed_origin_bf16 import reduction
from n64_parts import Native


def main(ctx, output):
    native = Native()
    core = ctx.full.core
    state = ctx.retained['narrow_control']
    stream = dedicated(ctx)
    core._sides[stream.cuda_stream] = dedicated(ctx)

    def captured(call):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            value = call()
        graph.replay()
        torch.cuda.synchronize()
        return graph, value

    def elapsed(graph, repeats=10):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeats

    records = []
    eviction = torch.zeros(64 << 20, device='cuda', dtype=torch.float32)
    for number, ((index, length), (x, weight)) in enumerate(state['samples'].items()):
        if length not in (192, 256):
            continue
        torch.save(dict(input=x.cpu(), weight=weight.cpu(), layer=index, length=length), output / f'source-{number}.pt')
        for count in (32, 128, 320, *range(512,min(len(x),1152)+1,32)):
            if count > len(x):
                continue
            a = x[:count].contiguous()
            config = reduction(core, (64, 2560), length, count)
            split = torch.full((1,), count, device=x.device, dtype=torch.int32)
            with ctx.full.workspace():
                base_graph, expected = captured(lambda: core.gemm(weight, a, length))
            expected = expected.clone()
            for name in native.variants:
                graph, actual = captured(lambda: native.project(a, weight, config, name))
                equal = torch.equal(expected.view(torch.int16), actual.view(torch.int16))
                row = dict(layer=index, length=length, rows=count, kernel=name, original_reduction=config, all_bits_equal=equal)
                if not equal:
                    torch.save(dict(source=f'source-{number}.pt', rows=count, expected=expected.cpu(), actual=actual.cpu(), record=row),
                               output / f'failure-{number}-{count}-{name}.pt')
                else:
                    hot = dict(base=[], candidate=[])
                    cold = dict(base=[], candidate=[])
                    for arm in ('base', 'candidate', 'candidate', 'base'):
                        selected = base_graph if arm == 'base' else graph
                        hot[arm].append(elapsed(selected))
                        eviction.add_(1)
                        cold[arm].append(elapsed(selected, 1))
                    ms = {label: statistics.median(v) for label, v in hot.items()}
                    cms = {label: statistics.median(v) for label, v in cold.items()}
                    row.update(milliseconds=ms, cold_milliseconds=cms, speedup=ms['base']/ms['candidate'],
                               cold_speedup=cms['base']/cms['candidate'])
                records.append(row)
                (output / 'operators.json').write_text(json.dumps(records, indent=2))
    result = dict(cases=len(records), all_bits_equal=all(r['all_bits_equal'] for r in records),
                  scope='N64,K2560 only; original K64 BF16 partials and ordered FP32 fold. Independent parts rather than serial CTA loop.')
    ctx.retained['n64_fixed_parallel'] = dict(native=native, operators=records, samples=state['samples'], result=result, source=str(output))
    return result
