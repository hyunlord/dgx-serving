#!/usr/bin/env python3
"""Serial throwaway warmup then one benchmark; raw stdout remains unchanged."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--warm',required=True);p.add_argument('--line-log',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--']else a.command
if not cmd:raise SystemExit('missing command')
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ.get('PATH','')
with Path(a.line_log).open('x')as side:
 def run(command,phase):
  start=time.time();print('C_INTERVAL '+json.dumps({'phase':phase,'edge':'start','epoch_seconds':start,'command':command}),flush=True)
  child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  assert child.stdout is not None
  for line in child.stdout:
   print(line,end='',flush=True);side.write(json.dumps({'epoch_seconds':time.time(),'phase':phase,'line':line.rstrip('\n')})+'\n');side.flush()
  rc=child.wait();print('C_INTERVAL '+json.dumps({'phase':phase,'edge':'end','epoch_seconds':time.time(),'exit_code':rc}),flush=True);return rc
 rc=run([a.warm,'--load','10'],'throwaway_warm')
 if rc:raise SystemExit(rc)
 raise SystemExit(run(cmd,'benchmark'))
