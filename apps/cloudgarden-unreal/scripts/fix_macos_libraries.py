#!/usr/bin/env python3
"""Complete packaged Unreal dylib closure from the installed engine, then ad-hoc sign."""
import pathlib,shutil,subprocess,sys
app=pathlib.Path(sys.argv[1]).resolve();engine=pathlib.Path('/Users/Shared/Epic Games/UE_5.8/Engine');dest=app/'Contents/MacOS'
stage=pathlib.Path(__file__).resolve().parents[1]/'Saved/StagedBuilds/Mac/Cloudgarden.app/Contents/UE'
if stage.exists():shutil.copytree(stage,app/'Contents/UE',dirs_exist_ok=True)
index={}
for p in engine.rglob('*.dylib'):
 index.setdefault(p.name,[]).append(p)
pending=[dest/'Cloudgarden'];seen=set();copied=[]
while pending:
 binary=pending.pop()
 if binary in seen:continue
 seen.add(binary)
 for line in subprocess.check_output(['otool','-L',str(binary)],text=True).splitlines()[1:]:
  dep=line.strip().split(' (')[0]
  if dep.startswith(('/System/','/usr/lib/')):continue
  name=pathlib.Path(dep).name
  if name==binary.name:continue
  target=dest/name
  if not target.exists():
   choices=index.get(name,[])
   if not choices:
    if 'weak)' in line:continue
    raise RuntimeError('Missing engine dependency '+dep)
   source=min(choices,key=lambda p:(0 if p.parent==engine/'Binaries/Mac' else 1,len(str(p))))
   shutil.copy2(source,target);copied.append(name)
  pending.append(target)
for binary in seen:
 if binary.name!='Cloudgarden':subprocess.run(['codesign','--force','--sign','-',str(binary)],check=True,capture_output=True)
subprocess.run(['codesign','--force','--deep','--sign','-',str(app)],check=True,capture_output=True)
subprocess.run(['codesign','--verify','--deep',str(app)],check=True)
print('Verified dylib closure:',len(seen),'binaries; copied',copied)
