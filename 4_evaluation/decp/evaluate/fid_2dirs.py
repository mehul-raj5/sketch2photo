import subprocess
import sys

def compute_fid(path1, path2):
    command = ['python', '-m', 'pytorch_fid', path1, path2, '--device', 'cuda:0']
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

if __name__ == '__main__':
    print(compute_fid('/data/cjh/codesketcher/result/ar_nothing_s2_scratch/b2a', '/data/cjh/codesketcher/data/ar/test/a'))