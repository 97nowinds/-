"""Restore only the missing entrance publisher; indoor publishers are untouched."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from feishu_app import save_credentials,load_credentials

CREDENTIALS=ROOT/'runtime'/'entrance-publisher.dpapi'
STATUS=ROOT/'runtime'/'entrance-publisher-status.json'


def status(state):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps({'state':state,'updated_at':time.time()}),encoding='utf-8')


def command(credentials,probe=False):
    import imageio_ffmpeg
    ip=credentials['ip']
    import ipaddress
    ipaddress.ip_address(ip)
    stream=credentials['stream']
    if stream not in ('stream1','stream2'):
        raise ValueError('invalid stream')
    source='rtsp://'+quote(credentials['username'],safe='')+':'+quote(credentials['password'],safe='')+'@'+ip+':554/'+stream
    args=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-loglevel','error',
          '-rtsp_transport','tcp','-i',source,'-map','0:v:0','-an']
    return args+(['-t','1','-f','null','-'] if probe else
                 ['-c:v','copy','-f','rtsp','-rtsp_transport','tcp','rtsp://127.0.0.1:18554/cam_entrance'])


def run():
    import psutil
    child=None
    try:
        while True:
            credentials=load_credentials(CREDENTIALS)
            child=subprocess.Popen(command(credentials),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            status('publisher_started')
            misses=0
            while child.poll() is None:
                time.sleep(5)
                try:
                    connected=any(c.status==psutil.CONN_ESTABLISHED and c.raddr and c.raddr.port==18554
                                  for c in psutil.Process(child.pid).net_connections('tcp'))
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    connected=False
                misses=0 if connected else misses+1
                status('publisher_connected' if connected else 'publisher_waiting')
                if misses>=4:
                    child.terminate()
                    child.wait(timeout=5)
            status('publisher_retrying')
            time.sleep(3)
    finally:
        if child and child.poll() is None:
            child.terminate()


def configure():
    import tkinter as tk
    from tkinter import ttk
    root=tk.Tk();root.title('恢复入口摄像头推流 — 本机隐藏输入');root.geometry('650x460')
    root.attributes('-topmost',True)
    frame=ttk.Frame(root,padding=22);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='入口摄像头连接',font=('Microsoft YaHei',17)).pack(anchor='w')
    ttk.Label(frame,text='当前集群等待 cam_entrance，但本机只推送了两路室内摄像头。').pack(anchor='w',pady=12)
    fields={}
    for key,label,value,hidden in [('ip','入口摄像头 IP','192.168.31.192',False),
                                 ('username','RTSP 摄像头账号（不是飞书或 SSH 账号）','admin',False),
                                 ('password','RTSP 摄像头密码','',True)]:
        ttk.Label(frame,text=label).pack(anchor='w',pady=(8,2))
        entry=ttk.Entry(frame,show='●' if hidden else '');entry.insert(0,value);entry.pack(fill='x');fields[key]=entry
    info=tk.StringVar(value='仅恢复入口推流；凭据由当前 Windows 用户加密保存。')
    ttk.Label(frame,textvariable=info,wraplength=600).pack(anchor='w',pady=18)
    def finished(message):
        info.set(message);button['state']='normal'
    def submit():
        values={k:v.get().strip() for k,v in fields.items()};values['stream']='stream1'
        if not all(values.values()):info.set('请填写摄像头账号和密码。');return
        button['state']='disabled';info.set('正在检查是否能读取入口画面……')
        def connect():
            try:
                probe=subprocess.run(command(values,probe=True),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW,timeout=15)
                if probe.returncode:
                    root.after(0,finished,'入口画面读取失败。请核对摄像头 RTSP 账号、密码及 RTSP 开关。');return
                # One supervisor per camera. Reconfiguration is handled by a clean relaunch.
                import psutil
                for process in psutil.process_iter(['name','cmdline']):
                    try:
                        args=process.info['cmdline'] or []
                        if '--run' in args and str(Path(__file__).resolve()) in args:
                            root.after(0,finished,'入口推流程序已运行，请先检查摄像头是否恢复。');return
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                save_credentials(values,CREDENTIALS)
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run'],cwd=ROOT,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                (ROOT/'runtime'/'entrance-publisher.pid').write_text(str(child.pid),encoding='ascii')
                root.after(0,finished,'入口画面读取成功，入口推流程序已启动；等待集群接收后监控页会自动恢复。')
            except subprocess.TimeoutExpired:
                root.after(0,finished,'读取入口画面超时，请检查摄像头和账号设置。')
            except Exception:
                root.after(0,finished,'本机启动失败，请检查环境；凭据未写入日志。')
        threading.Thread(target=connect,daemon=True).start()
    button=ttk.Button(frame,text='验证画面并恢复入口推流',command=submit);button.pack(anchor='w')
    root.mainloop()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    run() if args.run else configure()
