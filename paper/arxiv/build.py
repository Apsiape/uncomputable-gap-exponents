"""Build a standalone manuscript in a specified output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True)
args=parser.parse_args()
source=Path(__file__).resolve().parent
out=Path(args.output_dir).resolve()
if out==source:
    raise ValueError('Choose a separate build directory')
out.mkdir(parents=True,exist_ok=True)
for path in sorted(source.glob('*.tex')) + [source/'references.bib']:
    shutil.copyfile(path,out/path.name)
commands=[['pdflatex','-interaction=nonstopmode',
           '-halt-on-error','-file-line-error','main.tex'],
          ['bibtex','main']]
commands += [commands[0],commands[0]]
for index,command in enumerate(commands,1):
    done=subprocess.run(command,cwd=out,capture_output=True,text=True,
                        encoding='utf-8',errors='replace',timeout=180)
    (out/('pass-%d.txt'%index)).write_text(done.stdout+done.stderr,encoding='utf-8')
    if done.returncode:
        raise RuntimeError('Build command failed; inspect its pass log')
log=(out/'main.log').read_text(encoding='utf-8',errors='replace')
blg=(out/'main.blg').read_text(encoding='utf-8',errors='replace')
issues=re.findall(r'^(?:LaTeX(?: Font)?|Package \S+|Class \S+) Warning[^\n]*|'
                  r'^(?:Overfull|Underfull) \\[hv]box[^\n]*|'
                  r'^.*undefined references.*$',log,re.M)
bib_issues=re.findall(r'^Warning--.*$|^I couldn.t open.*$',blg,re.M)
assert not issues, issues
assert not bib_issues, bib_issues
summary=dict(status='PASS',latex_warnings=0,bibtex_warnings=0,
             undefined_references=0,overfull_boxes=0,underfull_boxes=0,
             pdf_sha256=hashlib.sha256((out/'main.pdf').read_bytes()).hexdigest())
(out/'build-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))
