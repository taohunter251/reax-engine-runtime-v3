#!/usr/bin/env python3
"""Rebuild and verify the three frozen CUDA binaries without model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='/usr/local/cuda-13.0/bin/nvcc')
    parser.add_argument('--output', type=Path, default=ROOT/'build')
    parser.add_argument('--shipped', type=Path,
                        help='Optional directory containing the image\'s hash-named cubins')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    version = subprocess.check_output([args.nvcc,'--version'],text=True)
    if 'V13.0.88' not in version:
        raise SystemExit('Byte-identical verification requires nvcc V13.0.88')
    args.output.mkdir(parents=True,exist_ok=True)
    results = []
    for item in manifest['kernels']:
        source = ROOT/item['source']
        if digest(source) != item['source_sha256']:
            raise SystemExit('Source checksum mismatch: '+item['source'])
        if 'header' in item and digest(ROOT/item['header']) != item['header_sha256']:
            raise SystemExit('Header checksum mismatch: '+item['header'])
        target = args.output/item['binary_name']
        command = [args.nvcc,*manifest['toolchain']['compiler_flags']]
        if 'header' in item:
            command += ['-I'+str((ROOT/item['header']).parent)]
        command += [str(source),'-o',str(target)]
        with (args.output/(item['name']+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual = digest(target)
        exact = actual == item['binary_sha256']
        if args.shipped is not None:
            shipped = args.shipped/(item['binary_sha256']+'.cubin')
            exact = exact and digest(shipped) == actual and shipped.read_bytes() == target.read_bytes()
        row = {'kernel':item['name'],'sha256':actual,'byte_identical':exact}
        results.append(row)
        (args.output/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if not exact:
            raise SystemExit('Rebuild differs from the frozen image binary')


if __name__ == '__main__':
    main()
