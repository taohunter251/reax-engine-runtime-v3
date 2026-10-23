#!/usr/bin/env python3
"""Rebuild the six inherited CUDA objects, without GPU access or model weights."""
import argparse,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nvcc',required=True);p.add_argument('--output',type=Path,default=R/'build/inherited');p.add_argument('--shipped',type=Path);a=p.parse_args()
if 'V13.4.92' not in subprocess.check_output([a.nvcc,'--version'],text=True):raise SystemExit('Requires nvcc V13.4.92')
a.output.mkdir(parents=True,exist_ok=True);rows=[]
for k in json.loads((R/'inherited-manifest.json').read_text())['kernels']:
 src=R/k['source'];assert hashlib.sha256(src.read_bytes()).hexdigest()==k['source_sha256'],k['source']
 out=a.output/k['binary_name'];cmd=[a.nvcc,*k['compiler_flags'],'-o',str(out),str(src)]
 with (a.output/(k['name']+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 h=hashlib.sha256(out.read_bytes()).hexdigest();exact=h==k['binary_sha256']
 if a.shipped is not None:exact=exact and out.read_bytes()==(a.shipped/k['binary_name']).read_bytes()
 row=dict(kernel=k['name'],sha256=h,byte_identical=exact);rows.append(row);(a.output/'verification.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 if not exact:raise SystemExit('Rebuild does not match shipped binary: '+k['name'])
