from pathlib import Path
import json,torch
from ordered_tma_table_native import Native
torch.empty(1,device='cuda');native=Native();domain=native.domain_check();records=[]
w=torch.ones((512,2560),device='cuda',dtype=torch.bfloat16)
for m in (129,1024):
    x=torch.ones((m,2560),device='cuda',dtype=torch.bfloat16)
    expected=torch.full((m,256),2560*2560,device='cuda',dtype=torch.bfloat16)
    for name in native.variants:
        actual=native.product(x,w,name);torch.cuda.synchronize()
        equal=torch.equal(actual.view(torch.int16),expected.view(torch.int16))
        assert equal,(m,name)
        records.append(dict(rows=m,kernel=name,all_bits_equal=True))
print(json.dumps(dict(cases=len(records),all_bits_equal=True,domain=domain)))
