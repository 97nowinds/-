"""Windows resident owner for transport, camera publishers and local services.

Only owned child process trees are restarted. Source URLs and process command
lines never enter logs or status output. Camera credentials are DPAPI protected.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import psutil
from feishu_app import load_credentials


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.4):return True
    except OSError:return False


def source_url(camera):
    import ipaddress
    ipaddress.ip_address(camera['ip'])
    if camera['stream'] not in ('Streaming/Channels/101','Streaming/Channels/102','stream1','stream2'):
        raise ValueError('Unsupported camera stream')
    return 'rtsp://'+quote(camera['username'],safe='')+':'+quote(camera['password'],safe='')+'@'+camera['ip']+':554/'+camera['stream']


def publisher_command(camera,port,probe=False):
    import imageio_ffmpeg
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error',
             '-rtsp_transport','tcp','-i',source_url(camera),'-map','0:v:0','-an']
    if probe:return command+['-t','1','-f','null','-']
    if camera['id'] not in ('cam_1','cam_2','cam_entrance'):raise ValueError('Unknown camera')
    return command+['-c:v','copy','-progress','pipe:1','-stats_period','5',
                    '-f','rtsp','-rtsp_transport','tcp','rtsp://127.0.0.1:'+str(port)+'/'+camera['id']]


class Child:
    def __init__(self,name,command,cwd,logger,env=None,port=None,progress=False):
        self.name,self.command,self.cwd,self.log=name,command,cwd,logger
        self.env,self.port,self.progress=env,port,progress
        self.process=None;self.started=0.;self.next_start=0.;self.last_frame=0.;self.frames=0
        self.failure_count=0;self.state='waiting';self.restarts=0

    def stop(self,reason):
        child=self.process
        if child is not None and child.poll() is None:
            try:
                parent=psutil.Process(child.pid)
                # Child handles originate only from this owner, never a port scan.
                children=parent.children(recursive=True)
                for p in reversed(children):
                    try:p.terminate()
                    except psutil.NoSuchProcess:pass
                parent.terminate()
                _,alive=psutil.wait_procs([parent,*children],timeout=3)
                for p in alive:
                    try:p.kill()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
        self.process=None;self.next_start=time.monotonic()+5;self.state='retrying'
        self.log.info('%s stopped: %s',self.name,reason)

    def ensure(self,ready=True):
        now=time.monotonic()
        if self.process is not None and self.process.poll() is not None:
            self.log.warning('%s exited: code=%s',self.name,self.process.returncode)
            self.process=None;self.next_start=now+5;self.state='retrying'
        if self.process is not None:return
        if not ready:self.state='waiting_dependency';return
        if self.port and listening(self.port):self.state='external_listener';return
        if now<self.next_start:return
        try:
            output=subprocess.PIPE if self.progress else subprocess.DEVNULL
            self.process=subprocess.Popen(self.command,cwd=self.cwd,env=self.env,stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            self.started=self.last_frame=now;self.frames=0;self.restarts+=1;self.state='starting'
            self.log.info('%s started pid=%d',self.name,self.process.pid)
            if self.progress:
                threading.Thread(target=self.read_progress,args=(self.process,),daemon=True).start()
        except Exception as e:
            self.state='start_failed';self.next_start=now+5
            self.log.error('%s start failed: %s',self.name,type(e).__name__)

    def read_progress(self,child):
        for raw in child.stdout:
            if child is not self.process:break
            if raw.startswith(b'frame='):
                try:
                    frames=int(raw.split(b'=',1)[1])
                    if frames>self.frames:self.frames=frames;self.last_frame=time.monotonic();self.state='streaming'
                except ValueError:pass

    def public(self):
        return {'state':self.state,'pid':self.process.pid if self.process and self.process.poll() is None else None,
                'starts':self.restarts,'frames':self.frames,
                'frame_progress_age_seconds':round(time.monotonic()-self.last_frame,1) if self.progress and self.process else None}


def keep_awake_on_ac(enabled):
    if not enabled:return
    class Power(ctypes.Structure):
        _fields_=[('ac',wintypes.BYTE),('battery',wintypes.BYTE),('percent',wintypes.BYTE),
                  ('status',wintypes.BYTE),('life',wintypes.DWORD),('full',wintypes.DWORD)]
    power=Power()
    plugged=ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) and power.ac==1
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000|(1 if plugged else 0))


def own_process_tree():
    """Close all descendant processes if the scheduled owner exits or crashes."""
    class Basic(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),
                  ('flags',wintypes.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
                  ('active',wintypes.DWORD),('affinity',ctypes.c_size_t),
                  ('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
    class Limits(ctypes.Structure):
        _fields_=[('basic',Basic),('io',ctypes.c_uint64*6),('process_memory',ctypes.c_size_t),
                  ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.restype=wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    handle=kernel.CreateJobObjectW(None,None)
    limits=Limits();limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not handle or not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle,kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/resident_services.json')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    import msvcrt
    lock=(runtime/'resident-services.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:return
    job_handle=own_process_tree()
    logger=logging.getLogger('resident');logger.setLevel(logging.INFO)
    handler=RotatingFileHandler(runtime/'resident-services.log',maxBytes=2*1024*1024,backupCount=4,encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));logger.addHandler(handler)
    key=str(Path(config['ssh_identity']).expanduser());ssh=str(Path('C:/Windows/System32/OpenSSH/ssh.exe'))
    proxy='ssh -i "'+key+'" -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -W %h:%p '+config['ssh_jump']
    tunnel=Child('cluster_tunnel',[ssh,'-NT','-i',key,'-o','BatchMode=yes','-o','IdentitiesOnly=yes',
        '-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
        '-o','ExitOnForwardFailure=yes','-o','ProxyCommand='+proxy,
        '-L','127.0.0.1:'+str(config['api_port'])+':127.0.0.1:5000',
        '-L','127.0.0.1:'+str(config['ingest_port'])+':127.0.0.1:8554',config['ssh_target']],ROOT,logger,port=config['api_port'])
    import os
    env=dict(os.environ);env.update(LAB_MANAGEMENT_ENABLED='1' if config['management_enabled'] else '0',
                                  LAB_HARDWARE_PORT=config['hardware_port'],LAB_HARDWARE_AUTO='0',PYTHONUNBUFFERED='1')
    python=str(ROOT/'.venv/Scripts/python.exe');api='http://127.0.0.1:'+str(config['api_port'])
    frontend=Child('frontend',[python,str(ROOT/'frontend_server.py'),'--host','0.0.0.0','--port',
        str(config['frontend_port']),'--api',api],ROOT,logger,env=env,port=config['frontend_port'])
    handoff=Child('map_handoff',[python,str(ROOT/'map_handoff.py'),'--api',api,'--port',str(config['handoff_port']),
        '--credentials-file',str(runtime/'handoff-credentials.json'),'--sample-archive',str(runtime/'handoff-samples/archive')],ROOT,logger,port=config['handoff_port'])
    children=[tunnel,frontend,handoff];publishers={};credential_revision=None
    opener=build_opener(ProxyHandler({}));api_failures=0;transport_failures=0
    logger.info('Resident owner started; all output is metadata only')
    try:
        while True:
            keep_awake_on_ac(config.get('keep_awake_on_ac',True))
            tunnel.ensure();frontend.ensure()
            handoff.ensure(ready=(runtime/'handoff-credentials.json').exists())
            credentials=runtime/'resident-camera-sources.dpapi'
            revision=credentials.stat().st_mtime_ns if credentials.exists() else None
            if revision!=credential_revision:
                try:
                    cameras=load_credentials(credentials)['cameras'] if revision else []
                    for child in publishers.values():child.stop('credentials updated')
                    publishers={c['id']:Child('publisher_'+c['id'],publisher_command(c,config['ingest_port']),ROOT,logger,progress=True) for c in cameras}
                    credential_revision=revision
                except Exception as e:logger.error('Protected camera configuration unavailable: %s',type(e).__name__)
            # Recover the already configured entrance without requiring re-entry.
            if not publishers and (runtime/'entrance-publisher.dpapi').exists():
                c=load_credentials(runtime/'entrance-publisher.dpapi');c['id']='cam_entrance'
                publishers={'cam_entrance':Child('publisher_cam_entrance',publisher_command(c,config['ingest_port']),ROOT,logger,progress=True)}
            for child in publishers.values():
                child.ensure(ready=listening(config['ingest_port']))
                if child.process and time.monotonic()-child.last_frame>config['camera_stall_seconds']:
                    child.stop('no frame progress')
            state={'updated_at':time.time(),'owner_pid':os.getpid(),'services':{c.name:c.public() for c in children},
                   'publishers':{k:v.public() for k,v in publishers.items()},
                   'missing_camera_credentials':[k for k in ['cam_1','cam_2','cam_entrance'] if k not in publishers]}
            try:
                with opener.open(api+'/api/state',timeout=8) as response:body=json.load(response)
                state['backend']={'available':True,'cameras':[{'id':c['id'],'status':c['status'],
                    'capture_age_seconds':c.get('capture_age_seconds'),'tracker_error':bool(c.get('tracker_error'))} for c in body['cameras']]}
                api_failures=0
                transport_failures=0
            except Exception:
                api_failures+=1;state['backend']={'available':False,'consecutive_failures':api_failures}
                # A slow state/model request is not evidence that the RTSP/SSH tunnel is broken.
                try:
                    with opener.open(api+'/api/health',timeout=3) as response:
                        if response.status!=200:raise ValueError('transport health')
                    transport_failures=0
                    state['backend']['transport_available']=True
                except Exception:
                    transport_failures+=1
                    state['backend']['transport_available']=False
                if transport_failures>=3 and tunnel.process:
                    tunnel.stop('transport health failed three times');transport_failures=0
            for child in [frontend,handoff]:
                if child.process and time.monotonic()-child.started>30:
                    port=config['frontend_port'] if child is frontend else config['handoff_port']
                    path='/health'
                    try:
                        with opener.open('http://127.0.0.1:'+str(port)+path,timeout=8) as response:response.read(64)
                        child.failure_count=0;child.state='running'
                    except Exception:
                        child.failure_count+=1
                        if child.failure_count>=3:child.stop('HTTP health failed');child.failure_count=0
            remote_cameras=state.get('backend',{}).get('cameras',[])
            state['healthy']=(state['backend']['available'] and len(remote_cameras)==3
                and all(c['status']=='running' and c['capture_age_seconds'] is not None and c['capture_age_seconds']<4 for c in remote_cameras)
                and not state['missing_camera_credentials']
                and all(c.process and c.process.poll() is None for c in [*children,*publishers.values()]))
            temp=runtime/'resident-status.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(runtime/'resident-status.json')
            time.sleep(config['poll_seconds'])
    finally:
        for child in [*publishers.values(),*children]:child.stop('owner exiting')
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:
        main()
    except Exception:
        # A scheduled pythonw owner has no console; keep startup failures visible.
        import traceback
        (ROOT/'runtime').mkdir(exist_ok=True)
        with (ROOT/'runtime/resident-startup-error.log').open('a',encoding='utf-8') as output:
            traceback.print_exc(file=output)
        raise
