#!/usr/bin/env python3
"""Bundle the installed Node runtime and non-system dylibs for local Unreal use."""
import hashlib, pathlib, re, shutil, subprocess, sys
out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/cloudgarden-unreal-runtime/node-runtime')
(out/'bin').mkdir(parents=True,exist_ok=True);(out/'lib').mkdir(exist_ok=True)
node=pathlib.Path(shutil.which('node')).resolve(); pending=[node]; seen={}; edits=[]
def run(*args):return subprocess.check_output(args,text=True)
def expand(path,base):return path.replace('@loader_path',str(base.parent)).replace('@executable_path',str(node.parent))
while pending:
 source=pending.pop().resolve()
 if source in seen:continue
 target=out/('bin/node' if source==node else 'lib/'+source.name)
 if target.exists() and source!=node and hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():
  raise RuntimeError('Conflicting library name: '+source.name)
 shutil.copy2(source,target);seen[source]=target
 deps=[line.strip().split(' (')[0] for line in run('otool','-L',str(source)).splitlines()[1:]]
 rpaths=re.findall(r'cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.*?) \(offset',run('otool','-l',str(source)))
 for dep in deps:
  if dep.startswith(('/usr/lib/','/System/')):continue
  if dep.startswith('@rpath/'):
   candidates=[pathlib.Path(expand(p,source))/dep[len('@rpath/'): ] for p in rpaths]
   actual=next((p for p in candidates if p.exists()),None)
   if actual is None:raise RuntimeError('Unresolved '+dep+' in '+str(source))
  else:actual=pathlib.Path(expand(dep,source))
  actual=actual.resolve()
  if actual==source:continue
  pending.append(actual)
  replacement=('@loader_path/../lib/' if source==node else '@loader_path/')+actual.name
  edits.append((target,dep,replacement))
for target,old,new in edits:subprocess.run(['install_name_tool','-change',old,new,str(target)],check=True,capture_output=True)
for target in seen.values():subprocess.run(['codesign','--force','--sign','-',str(target)],check=True,capture_output=True)
print(run(str(out/'bin/node'),'--version').strip(),len(seen)-1,'bundled libraries',out)
