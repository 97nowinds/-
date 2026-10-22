"""Run on lnx as the existing project user; adopt the running API/relay.

Environment snapshots stay on that host with mode 0600 and are never printed.
No model, mapping, camera input or notification policy is changed here.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

ROOT=Path('/cluster/home/xqyi/lab-safety-host/camera')
PYTHON='/cluster/home/opt/anaconda3/envs/omat25/bin/python'


def active(name):
    return subprocess.run(['systemctl','--user','is-active','--quiet',name]).returncode==0


def find_process(predicate):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=[s for s in (p/'cmdline').read_bytes().decode().split(chr(0)) if s]
            cwd=(p/'cwd').resolve()
            if predicate(args,cwd):return int(p.name),args,cwd
        except (OSError,UnicodeError):pass
    return None


def install():
    state=ROOT/'runtime/resident';state.mkdir(exist_ok=True,mode=0o700);state.chmod(0o700)
    launcher=state/'exec.py'
    launcher.write_text("from pathlib import Path\nimport os,sys,json\nd=json.loads(Path(sys.argv[1]).read_text());os.chdir(d['cwd']);os.execve(d['argv'][0],d['argv'],d['env'])\n")
    health=state/'health.py'
    health.write_text('''import time,subprocess,urllib.request,json
from pathlib import Path
failures=0
while True:
    try:
        with urllib.request.urlopen('http://127.0.0.1:5000/api/health',timeout=3) as response:
            if response.status!=200:raise ValueError('health status')
        failures=0
    except Exception:
        failures+=1
        if failures>=3:
            print('API health failed three times; restarting project service',flush=True)
            subprocess.run(['systemctl','--user','restart','lab-safety-api.service'])
            failures=0;time.sleep(30)
    time.sleep(10)
''')
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    targets=[('lab-safety-relay.service',lambda a,c:a and Path(a[0]).name=='mediamtx'),
             ('lab-safety-api.service',lambda a,c:c==ROOT and 'app.py' in a)]
    old=[]
    for unit,predicate in targets:
        config=state/(unit+'.json');found=find_process(predicate)
        if found:
            pid,args,cwd=found
            environment=dict(x.split('=',1) for x in Path('/proc/'+str(pid)+'/environ').read_bytes().decode().split(chr(0)) if '=' in x)
            environment['PYTHONUNBUFFERED']='1'
            args[0]=str((cwd/args[0]).resolve()) if not Path(args[0]).is_absolute() else args[0]
            config.write_text(json.dumps({'argv':args,'cwd':str(cwd),'env':environment}));config.chmod(0o600)
            if not active(unit):old.append(pid)
        elif not config.exists():raise RuntimeError('Cannot provision '+unit+' without a running process or prior protected configuration')
        extra='After=lab-safety-relay.service\nWants=lab-safety-relay.service\n' if 'api' in unit else ''
        (units/unit).write_text('[Unit]\nDescription=Laboratory resident '+unit+'\nStartLimitIntervalSec=0\n'+extra+
            '\n[Service]\nType=simple\nUMask=0077\nExecStart='+PYTHON+' '+str(launcher)+' '+str(config)+
            '\nRestart=always\nRestartSec=5\nTimeoutStopSec=20\nLogRateLimitIntervalSec=30s\nLogRateLimitBurst=100\n'+
            '\n[Install]\nWantedBy=default.target\n')
    (units/'lab-safety-health.service').write_text('[Unit]\nDescription=Laboratory API health recovery\nAfter=lab-safety-api.service\nStartLimitIntervalSec=0\n'+
        '\n[Service]\nType=simple\nExecStart='+PYTHON+' -u '+str(health)+'\nRestart=always\nRestartSec=5\n'+
        '\n[Install]\nWantedBy=default.target\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['loginctl','enable-linger',os.environ['USER']],check=True)
    for pid in old:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+10
    while any(Path('/proc/'+str(pid)).exists() for pid in old) and time.monotonic()<deadline:time.sleep(.2)
    for unit in ['lab-safety-relay.service','lab-safety-api.service','lab-safety-health.service']:
        subprocess.run(['systemctl','--user','enable','--now',unit],check=True)
    print(json.dumps({'units':{u:active(u) for u in ['lab-safety-relay.service','lab-safety-api.service','lab-safety-health.service']},
                      'protected_configuration_directory':str(state)}))


if __name__=='__main__':install()
