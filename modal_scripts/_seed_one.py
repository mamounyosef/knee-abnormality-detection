import subprocess, sys
FILES = ['D:\\My Projects\\knee_abnormality_detection\\data\\pretrained\\medimageinsight_davit_huge.safetensors', 'D:\\My Projects\\knee_abnormality_detection\\data\\pretrained\\MRI_CORE_vitb.pth', 'D:\\My Projects\\knee_abnormality_detection\\data\\pretrained\\MRI_CORE_vitb_mae.pth']
for f in FILES:
    name = f.replace(chr(92), '/').rsplit('/', 1)[-1]
    r = subprocess.run([sys.executable, '-m', 'modal', 'volume', 'put',
                        'knee-runs', f, 'pretrained/' + name],
                       capture_output=True, text=True, errors='replace')
    ok = r.returncode == 0 or 'already exists' in (r.stdout + r.stderr)
    print(('  ok   ' if ok else '  FAIL ') + name, flush=True)
    if not ok:
        print((r.stdout + r.stderr)[-400:])
