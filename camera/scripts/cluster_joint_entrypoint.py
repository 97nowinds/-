"""Install beside the two isolated service directories; private.json must be mode 0600.
Runtime template copied from the validated cluster deployment on 2026-10-26.
"""
import os,json,sys,runpy,http.client,socket
from pathlib import Path
base=Path(__file__).resolve().parent
role=sys.argv[1];sys.argv=[role]+sys.argv[2:]
data=json.loads((base/'private.json').read_text())
os.environ.update(data['envs'][role]);os.environ.update(CUDA_VISIBLE_DEVICES='0',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
import torch
import faulthandler,signal
faulthandler.register(signal.SIGUSR1,all_threads=True)
def thread_inventory(sig, frame):
 import collections,traceback,threading
 counts=collections.Counter()
 for ident,stack in sys._current_frames().items():
  frames=[f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_stack(stack) if 'joint-migration' in f.filename]
  counts[tuple(frames[-6:])]+=1
 print('THREAD_INVENTORY '+json.dumps([[n,list(frames)] for frames,n in counts.most_common()]),file=sys.stderr,flush=True)
signal.signal(signal.SIGUSR2,thread_inventory)
torch.set_num_threads(2)
if role=='8765':
 from sqlite_connection_guard import install as install_sqlite_guard
 install_sqlite_guard()
 original=http.client.HTTPConnection.connect
 def connect(self):
  if self.host=='127.0.0.1' and self.port==5177:
   self.sock=socket.create_connection(('127.0.0.1',25177),self.timeout,self.source_address)
  else:original(self)
 http.client.HTTPConnection.connect=connect
os.chdir(base/role);sys.path.insert(0,str(base/role))
if role=='15002':runpy.run_path(str(base/role/'app.py'),run_name='__main__')
else:runpy.run_module('member_handoff_service.server',run_name='__main__')

