"""Run every finite check sequentially with one numerical-library thread."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,json,subprocess,sys,time
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
scripts=sys.argv[1:] or ['finite_size_runner.py','check_wall_layer.py','check_wall_layer_noO.py',
         'check_counter_perturbation.py','check_film_counter.py',
         'check_film_classical_layer.py','check_nested_limits.py','check_critical_ruler.py','check_bulk_broadcast.py','check_bulk_independent.py']
previous=HERE/'all_checks_summary.json'
records=json.loads(previous.read_text())['checks'] if sys.argv[1:] and previous.exists() else []
for name in scripts:
    print('Running '+name,flush=True)
    start=time.monotonic(); peak=0
    output=HERE/(Path(name).stem+'_output.txt')
    with output.open('w',encoding='utf-8') as out:
        proc=subprocess.Popen([sys.executable,'-u',name],cwd=HERE,stdout=out,stderr=subprocess.STDOUT)
        monitor=psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                processes=[monitor]+monitor.children(recursive=True)
                memory=sum(p.memory_info().rss for p in processes if p.is_running())
                peak=max(peak,memory)
                if memory>=1000000000:
                    for p in reversed(processes): p.kill()
                    raise RuntimeError('Memory bound exceeded by '+name)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic()-start>2400:
                proc.kill(); raise RuntimeError('Time bound exceeded by '+name)
            time.sleep(.2)
    text=output.read_text(encoding='utf-8')
    if proc.returncode or 'FAIL' in text or 'Traceback' in text:
        raise RuntimeError(name+' failed; inspect recorded output')
    records.append(dict(script=name,status='PASS',seconds=round(time.monotonic()-start,3),peak_bytes=peak))
    print(name+' PASS',flush=True)
summary=dict(status='PASS',single_threaded=True,memory_limit_bytes=1000000000,checks=records,
             numerical_caveat='Floating-point eigenvalues are consistency checks, not interval proofs.')
(HERE/'all_checks_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print('ALL FINITE CHECKS PASS',flush=True)
