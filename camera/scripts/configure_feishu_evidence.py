"""Masked three-group configuration and explicit synthetic image connectivity test."""
from io import BytesIO
import json
from pathlib import Path
import secrets
import sys
import threading
import tkinter as tk
from tkinter import ttk

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from feishu_app import FeishuAppClient,FeishuError,EVIDENCE_CREDENTIALS,GROUP_CREDENTIALS,save_credentials,load_credentials
from alert_evidence import group_evidence_payload
from management_notifications import send_feishu,ManagementNotifications

STATUS=ROOT/'runtime'/'feishu-evidence-setup-status.json'


def status(state,detail=None):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps({'state':state,'detail':detail},ensure_ascii=False),encoding='utf-8')


def main():
    root=tk.Tk();root.title('防护告警证据图片与三个飞书群接入');root.geometry('760x590')
    root.attributes('-topmost',True)
    container=ttk.Frame(root,padding=20);container.pack(fill='both',expand=True)
    ttk.Label(container,text='群消息证据图片配置',font=('Microsoft YaHei',17)).pack(anchor='w')
    ttk.Label(container,text='应用仅用于上传图片；不查询手机号/邮箱，不发送私聊，不启用电话。').pack(anchor='w',pady=10)
    tabs=ttk.Notebook(container);tabs.pack(fill='x',pady=8)
    entries={}
    specs=[('app','图片上传应用',[('app_id','App ID（cli_开头）',False),('app_secret','App Secret',True)])]
    specs += [(key,label,[('webhook','群自定义机器人完整 Webhook',True),('secret','签名密钥（未启用签名时留空）',True)])
              for key,label in [('experimenters','实验人员群'),('managers','实验负责人群'),('administrators','实验室管理人员群')]]
    for name,label,fields in specs:
        frame=ttk.Frame(tabs,padding=15);tabs.add(frame,text=label);entries[name]={}
        for key,text,hidden in fields:
            ttk.Label(frame,text=text).pack(anchor='w',pady=(8,4))
            entry=ttk.Entry(frame,width=80,show='●' if hidden else '');entry.pack(fill='x');entries[name][key]=entry
    ttk.Label(container,text='请填写四个标签页。下方按钮会上传一张明确标注“合成测试”的图片，\n并向三个已填写的群发送图文测试消息；不上传真人画面。').pack(anchor='w',pady=15)
    info=tk.StringVar(value='等待输入。Webhook、密钥和应用凭证均加密保存在本机。')
    ttk.Label(container,textvariable=info,wraplength=700).pack(anchor='w',pady=10)
    def done(ok,message):
        info.set(message);button['state']='normal'
        if ok:
            for fields in entries.values():
                for field in fields.values():field.delete(0,'end')
    def submit():
        values={group:{key:entry.get().strip() for key,entry in fields.items()} for group,fields in entries.items()}
        if not all(values['app'].values()) or any(not values[k]['webhook'] for k in values if k!='app'):
            info.set('请填写应用凭证和三个群的Webhook。');return
        button['state']='disabled';info.set('正在上传合成图片并向三个群发送测试……');status('connecting')
        def connect():
            delivered=[]
            try:
                from PIL import Image,ImageDraw
                picture=Image.new('RGB',(900,480),(235,241,246));draw=ImageDraw.Draw(picture)
                draw.text((40,50),'SYNTHETIC CONNECTIVITY TEST - NO REAL PERSON',(20,50,80))
                draw.rectangle((160,140,550,410),outline=(230,70,30),width=5)
                draw.text((180,170),'Example evidence bounding box',(20,50,80))
                output=BytesIO();picture.save(output,format='JPEG')
                client=FeishuAppClient(values['app']);image_key=client.upload_image(output.getvalue())
                for audience in ('experimenters','managers','administrators'):
                    bot=values[audience]
                    send_feishu(bot['webhook'],group_evidence_payload(
                        '实验室图文通知连接测试（合成数据）。\n图片为程序生成的测试图，不代表真人违规，不是火焰警报。',
                        image_key,bot['secret'],title='实验室图文连接测试｜合成数据'))
                    delivered.append(audience)
                old=load_credentials(GROUP_CREDENTIALS) if GROUP_CREDENTIALS.exists() else {}
                credentials={**values['app'],'groups':{k:v for k,v in values.items() if k!='app'},
                    'management_admin_token':old.get('management_admin_token') or secrets.token_urlsafe(32),'phone_enabled':False}
                save_credentials(credentials,EVIDENCE_CREDENTIALS)
                config=json.loads((ROOT/'config'/'management.json').read_text(encoding='utf-8'))
                service=ManagementNotifications(ROOT/'runtime'/'management.sqlite3',config)
                try:
                    service.set_routes({'role':'admin','id':'local-evidence-setup'},[{
                        'id':audience,'transport':'webhook','labs':[config['lab_id']],
                        'event_types':['safety.no_mask','safety.no_lab_coat','safety.no_gloves'] if audience=='experimenters' else [],
                        'webhook_env':'LAB_FEISHU_'+audience.upper()+'_WEBHOOK',
                        'secret_env':'LAB_FEISHU_'+audience.upper()+'_SECRET'}
                        for audience in ('experimenters','managers','administrators')])
                finally:service.db.close()
                status('connected')
                root.after(0,done,True,'三个群图文测试均已被飞书接受，配置已加密保存。请确认三群收到测试图片，再重启通知前端加载配置。')
            except Exception as exc:
                message=str(exc) if isinstance(exc,FeishuError) else '本机配置或图片上传失败（'+type(exc).__name__+'）。'
                message += ' 已成功提交测试的群：'+(', '.join(delivered) or '无')
                status('failed',message);root.after(0,done,False,message)
        threading.Thread(target=connect,daemon=True).start()
    button=ttk.Button(container,text='验证图片上传，并向三个群发送合成图文测试',command=submit)
    button.pack(anchor='w',pady=12)
    status('awaiting_input');root.mainloop()


if __name__=='__main__':main()
