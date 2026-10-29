"""Run the finite checks sequentially with one numerical-library thread."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
scripts = ['check_penalty_lemma.py', 'check_dynamic_counter.py',
           'check_classical_layer.py', 'independent_checks.py']
for name in scripts:
    print('Running ' + name, flush=True)
    completed = subprocess.run([sys.executable, '-u', name], cwd=HERE,
                               capture_output=True, text=True, timeout=1200)
    (HERE / (Path(name).stem + '_output.txt')).write_text(
        completed.stdout + completed.stderr, encoding='utf-8')
    if completed.returncode or 'FAIL' in completed.stdout or completed.stderr:
        raise RuntimeError(name + ' failed; inspect its recorded output')
    print(completed.stdout.splitlines()[-1], flush=True)

penalty = json.loads((HERE / 'check_penalty_lemma.json').read_text())
assert all(v['min_ratio'] >= 1 - 1e-10 for v in penalty.values())
counter = json.loads((HERE / 'check_dynamic_counter.json').read_text())
for item in counter:
    bound = 4.0 ** (-item['k_bits']) / (64 * len(item['radices']) ** 2)
    assert item['gap'] >= bound - 1e-10
    assert item['gap'] <= item['pred_main_cycle_gap'] + 1e-10
    assert item['maxdeg_one_head'] <= 2 and item['multi_head_all_penalized']
classical = json.loads((HERE / 'check_classical_layer.json').read_text())
assert all(v['min_energy'] == -3*v['L'] and v['number_of_minimisers'] == 1
           for v in classical)
independent = json.loads((HERE / 'independent_checks.json').read_text())
summary = dict(
    python=platform.python_version(),
    status='PASS',
    counter_cases=len(counter),
    largest_counter_dimension=max(v['dim'] for v in counter),
    classical_sizes=[v['L'] for v in classical],
    random_penalty_instances=2000,
    exhaustive_penalty_instances=independent['penalty']['checked'],
    independent_counter_cases=len(independent['counter']),
    four_flag_assignments=independent['classical']['low_energy_enumerated'],
    numerical_caveat='Floating-point diagonalization is evidence, not an interval proof.'
)
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print('ALL FINITE CHECKS PASS', flush=True)
