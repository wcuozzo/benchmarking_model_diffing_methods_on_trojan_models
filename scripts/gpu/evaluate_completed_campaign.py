"""Verify a completed adapter, evaluate it, and seal separately backed-up results."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path('/workspace/nanda_reproduction')
name=sys.argv[1]
assert name==Path(name).name
run=root/'runs'/name
manifest=json.loads((run/'FINAL_COMPLETE.json').read_text())
for filename,digest in manifest['files'].items():
    with (run/filename).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest,filename
out=root/'runs'/('evaluation_'+name)
assert not out.exists(), f'Refusing to overwrite {out}'
out.mkdir()
subprocess.run([sys.executable,str(root/'scripts/audit_generations.py'),'--adapter',str(run),'--label',name,'--eval_sets',str(root/'audit/raw_box1/eval_sets.json'),'--out',str(out/'generations.json')],check=True)
(out/'source_model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={}
for p in out.iterdir():
    if p.is_file():
        with p.open('rb') as f: files[p.name]=hashlib.file_digest(f,'sha256').hexdigest()
(out/'FINAL_COMPLETE.tmp').write_text(json.dumps({'files':files},indent=2)+'\n')
(out/'FINAL_COMPLETE.tmp').replace(out/'FINAL_COMPLETE.json')
print('EVALUATION_SEALED',out,flush=True)
