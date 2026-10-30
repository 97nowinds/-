"""Own only the loopback SSH port for the site sample review service."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    import msvcrt
    lock=(ROOT/'runtime/face-adaptation-bridge.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:return
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from resident_services import own_process_tree
    owner=own_process_tree()
    config=json.loads((ROOT/'config/resident_services.json').read_text())
    key=str(Path(config['ssh_identity']).expanduser())
    proxy='ssh -i "'+key+'" -o BatchMode=yes -o ConnectTimeout=8 -W %h:%p '+config['ssh_jump']
    cmd=['C:/Windows/System32/OpenSSH/ssh.exe','-NT','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=8',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ExitOnForwardFailure=yes',
        '-o','ProxyCommand='+proxy,'-L','127.0.0.1:15003:127.0.0.1:15003',config['ssh_target']]
    while True:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        child.wait();time.sleep(5)


if __name__=='__main__':main()
