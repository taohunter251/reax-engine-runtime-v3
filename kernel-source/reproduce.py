"""Rebuild disclosed kernels and verify their exact shipped checksums."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc130',default='/usr/local/cuda-13.0/bin/nvcc')
    parser.add_argument('--nvcc134',required=True)
    parser.add_argument('--output',type=Path,default=Path('build'))
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--private-only',action='store_true')
    parser.add_argument('--runtime-directory',type=Path,
                        help='Install byte-verified rebuilt artifacts into a runtime checkout')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'manifest-v3.json').read_text())['kernels']
    if args.private_only: rows=[r for r in rows if not r.get('inherited')]
    def build(row):
        source=root/row['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
        compiler=row.get('compiler','nvcc')
        executable=('gcc' if compiler=='gcc' else args.nvcc134 if row['nvcc']=='13.4.92' else args.nvcc130)
        target=args.output.resolve()/(row['name'].replace('/','-')+('.so' if compiler=='gcc' else '.cubin'))
        command=[executable,*row['flags'],'-o',str(target),str(source)]
        with target.with_suffix('.log').open('w') as log:
            completed=subprocess.run(command,cwd=source.parent,stdout=log,stderr=subprocess.STDOUT)
        actual=hashlib.sha256(target.read_bytes()).hexdigest() if completed.returncode==0 else None
        result=dict(name=row['name'],command=command,returncode=completed.returncode,
                    sha256=actual,expected=row['binary_sha256'],byte_identical=actual==row['binary_sha256'])
        if result['byte_identical'] and args.runtime_directory is not None:
            for name in row['runtime_paths']:
                destination=args.runtime_directory/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target,destination)
        print(json.dumps(result),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results=list(executor.map(build,rows))
    (args.output/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
    if not all(r['byte_identical'] for r in results):
        raise SystemExit('Some outputs differ. Inspect logs and the documented original-path requirement.')

if __name__=='__main__':main()
