"""Own only the SSH transport for the isolated tracking and interaction services."""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import subprocess
import threading
import time
from datetime import datetime, timezone
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]


def receiver_healthy(opener, port=8765):
    """Exercise the event DB as well as the HTTP listener. Empty history is healthy."""
    try:
        with opener.open(f'http://127.0.0.1:{port}/api/demo/events',timeout=5) as response:
            body=json.load(response)
        return isinstance(body,dict) and isinstance(body.get('sessions'),list)
    except Exception:
        return False


def recover_receiver(config):
    """Distinguish a broken forward from a stalled owned remote unit."""
    key=str(Path(config['ssh_identity']).expanduser())
    proxy=('ssh -i "'+key+'" -o BatchMode=yes -o ConnectTimeout=8 -W %h:%p '+config['ssh_jump'])
    check=("if curl -fsS --max-time 5 http://127.0.0.1:8765/api/demo/events >/dev/null; "
           "then printf remote_healthy; else systemctl --user restart lab-joint-8765.service "
           "&& printf receiver_restarted; fi")
    try:
        result=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe','-T','-i',key,
            '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=8',
            '-o','ProxyCommand='+proxy,config['ssh_target'],check],
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            text=True,timeout=35,creationflags=subprocess.CREATE_NO_WINDOW)
        return result.stdout.strip() if result.returncode==0 else 'remote_unavailable'
    except (OSError,subprocess.TimeoutExpired):
        return 'remote_unavailable'


def command(config, validation=False):
    key = str(Path(config['ssh_identity']).expanduser())
    proxy = ('ssh -i "' + key + '" -o BatchMode=yes -o ConnectTimeout=8 '
             '-o ServerAliveInterval=15 -o ServerAliveCountMax=2 -W %h:%p ' + config['ssh_jump'])
    return ['C:/Windows/System32/OpenSSH/ssh.exe', '-NT', '-i', key,
            '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ProxyCommand=' + proxy,
            '-L', '127.0.0.1:' + ('25002' if validation else '15002') + ':127.0.0.1:5002',
            '-L', '127.0.0.1:' + ('28765' if validation else '8765') + ':127.0.0.1:8765',
            '-R', '127.0.0.1:25177:127.0.0.1:5177', config['ssh_target']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config/resident_services.json').read_text(encoding='utf-8-sig'))
    runtime = ROOT / 'runtime'
    import msvcrt
    lock = (runtime / 'cluster-joint-bridge.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return
    from resident_services import own_process_tree
    job = own_process_tree()
    logger = logging.getLogger('cluster-joint-bridge')
    logger.setLevel(logging.INFO)
    logger.addHandler(RotatingFileHandler(runtime / 'cluster-joint-bridge.log',
                      maxBytes=1024 * 1024, backupCount=2, encoding='utf-8'))
    opener=build_opener(ProxyHandler({}));last_recovery=0.;last_success=None
    while True:
        child = subprocess.Popen(command(config, args.validation), cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        logger.info('Started owned joint SSH bridge pid=%s', child.pid)
        def read_errors(stream,owned_child):
            for line in stream:
                logger.warning('SSH: %s', line.strip())
                if 'Too many open files' in line and owned_child.poll() is None:
                    # This child belongs to this bridge; a listening socket can still be unusable.
                    logger.warning('Recovering owned SSH bridge after socket exhaustion')
                    owned_child.terminate()
                    break
        threading.Thread(target=read_errors, args=(child.stderr,child), daemon=True).start()
        started=time.monotonic();failures=0
        while child.poll() is None:
            try:
                child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                pass
            healthy=receiver_healthy(opener,28765 if args.validation else 8765)
            failures=0 if healthy else failures+1
            if healthy:last_success=datetime.now(timezone.utc).isoformat()
            status={'updated_at':datetime.now(timezone.utc).isoformat(),'pid':child.pid,
                    'receiver_available':healthy,'consecutive_failures':failures,
                    'last_success_at':last_success,'automatic_recovery':True}
            if (not args.validation and failures>=3 and time.monotonic()-started>=60
                    and time.monotonic()-last_recovery>=120):
                last_recovery=time.monotonic()
                outcome=recover_receiver(config)
                status['recovery_action']=outcome
                logger.warning('Receiver event health failed three times; recovery=%s',outcome)
                # The source unit is checked independently before touching it.
                # Reopen only this owned tunnel after either transport or service recovery.
                if child.poll() is None:child.terminate()
            temp=runtime/'cluster-joint-bridge-status.tmp'
            temp.write_text(json.dumps(status,indent=2),encoding='utf-8')
            temp.replace(runtime/'cluster-joint-bridge-status.json')
        logger.warning('Owned joint SSH bridge exited code=%s; retrying', child.returncode)
        time.sleep(5)


if __name__ == '__main__':
    main()
