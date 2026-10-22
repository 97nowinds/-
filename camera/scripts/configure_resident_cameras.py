"""Masked local camera configuration, verified before DPAPI persistence."""
from pathlib import Path
import json
import subprocess
import sys
import threading
import tkinter as tk
import importlib
from tkinter import ttk

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from feishu_app import load_credentials,save_credentials
import resident_services


def main():
    path=ROOT/'runtime/resident-camera-sources.dpapi'
    pending=ROOT/'runtime/resident-camera-pending.dpapi'
    values=[{'id':'cam_1','ip':'192.168.31.191','username':'admin','password':'','stream':'Streaming/Channels/101'},
            {'id':'cam_2','ip':'192.168.31.190','username':'admin','password':'','stream':'Streaming/Channels/101'}]
    entrance=ROOT/'runtime/entrance-publisher.dpapi'
    if entrance.exists():values.append({'id':'cam_entrance',**load_credentials(entrance)})
    if path.exists():values=load_credentials(path)['cameras']
    elif pending.exists():values=load_credentials(pending)['cameras']
    window=tk.Tk();window.title('实验室长期值守 — 摄像头凭据安全保存');window.geometry('740x500');window.attributes('-topmost',True)
    body=ttk.Frame(window,padding=20);body.pack(fill='both',expand=True)
    ttk.Label(body,text='补齐摄像头自动恢复配置',font=('Microsoft YaHei',17)).pack(anchor='w')
    ttk.Label(body,text='凭据只在本机隐藏输入并加密保存；服务重启后无需再次输入。\n入口摄像头已有配置，已预填。不会向聊天或日志输出密码。').pack(anchor='w',pady=10)
    tabs=ttk.Notebook(body);tabs.pack(fill='x');fields={}
    for camera in values:
        page=ttk.Frame(tabs,padding=12);tabs.add(page,text=camera['id']);fields[camera['id']]={}
        for row,(key,label) in enumerate([('ip','摄像头 IP'),('username','账号'),('password','密码'),('stream','画面通道')]):
            ttk.Label(page,text=label).grid(row=row,column=0,sticky='w',padx=8,pady=6)
            entry=ttk.Entry(page,width=55,show='*' if key=='password' else '')
            entry.insert(0,camera.get(key,''));entry.grid(row=row,column=1,sticky='ew');fields[camera['id']][key]=entry
    result=ttk.Label(body,text='请填写 cam_1 和 cam_2 的密码，再点击验证。',wraplength=680);result.pack(anchor='w',pady=15)
    def submit():
        cameras=[{'id':cid,**{k:(e.get() if k=='password' else e.get().strip()) for k,e in entries.items()}} for cid,entries in fields.items()]
        if any(not c['password'] or not c['username'] for c in cameras):result.config(text='请补齐每路摄像头账号和密码。');return
        button.config(state='disabled');result.config(text='正在逐路验证真实视频，请稍候……')
        def work():
            verified=[]
            try:
                save_credentials({'cameras':cameras},pending)
                importlib.reload(resident_services)
                for camera in cameras:
                    proc=subprocess.run(resident_services.publisher_command(camera,18554,probe=True),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                        timeout=25,creationflags=subprocess.CREATE_NO_WINDOW)
                    if proc.returncode:raise ValueError(camera['id'])
                    verified.append(camera['id'])
                save_credentials({'cameras':cameras},path)
                (ROOT/'runtime/resident-camera-setup-status.json').write_text(json.dumps({'success':True,'verified':verified}))
                window.after(0,lambda:result.config(text='三路画面验证成功，凭据已加密保存。值守程序将在几秒内自动加载并推流。'))
            except Exception:
                failed=next((c['id'] for c in cameras if c['id'] not in verified),'unknown')
                window.after(0,lambda:result.config(text='验证未通过：'+failed+'。请检查该路账号、密码、网络和通道；已通过：'+', '.join(verified)))
            finally:window.after(0,lambda:button.config(state='normal'))
        threading.Thread(target=work,daemon=True).start()
    button=ttk.Button(body,text='验证三路画面并保存自动恢复配置',command=submit);button.pack(anchor='w')
    window.mainloop()


if __name__=='__main__':main()
