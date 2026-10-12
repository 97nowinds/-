"""Local group bot setup. Does not query users or enable telephone escalation."""
import json
from pathlib import Path
import secrets
import sys
import threading
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feishu_app import GROUP_CREDENTIALS, save_credentials, FeishuError
from management_notifications import ManagementNotifications, feishu_payload, send_feishu

STATUS = ROOT/'runtime'/'feishu-group-setup-status.json'


def status(state, detail=None):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({'state':state,'detail':detail},ensure_ascii=False), encoding='utf-8')


def main():
    root = tk.Tk()
    root.title('飞书群通知接入 — 本机安全输入')
    root.geometry('700x465')
    root.attributes('-topmost', True)
    frame = ttk.Frame(root,padding=24)
    frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='飞书群机器人通知',font=('Microsoft YaHei',17)).pack(anchor='w')
    ttk.Label(frame,text='在目标群：群设置 → 群机器人 → 自定义机器人，获取 Webhook。').pack(anchor='w',pady=12)
    ttk.Label(frame,text='Webhook 地址（隐藏输入）').pack(anchor='w')
    url = ttk.Entry(frame,show='●',width=80)
    url.pack(fill='x',pady=(3,12))
    ttk.Label(frame,text='签名密钥（仅在机器人启用签名校验时填写）').pack(anchor='w')
    secret = ttk.Entry(frame,show='●',width=80)
    secret.pack(fill='x',pady=(3,12))
    ttk.Label(frame,text='点击下方按钮，会向此机器人所在群发送一条明确标注的连接测试。\n测试成功后将通知路由切换为此群；不查询手机号/邮箱，不启用电话。').pack(anchor='w',pady=10)
    info=tk.StringVar(value='等待填写群机器人信息。输入内容加密保存在当前 Windows 用户下。')
    ttk.Label(frame,textvariable=info,wraplength=645).pack(anchor='w',pady=8)

    def finished(success,message):
        button['state']='normal'
        info.set(message)
        if success:
            url.delete(0,'end')
            secret.delete(0,'end')

    def submit():
        webhook, signing_secret = url.get().strip(),secret.get().strip()
        if not webhook:
            info.set('请填写 Webhook 地址。')
            return
        button['state']='disabled'
        info.set('正在向群发送连接测试……')
        status('connecting')
        def connect():
            try:
                send_feishu(webhook,feishu_payload('实验室通知系统：群机器人连接测试。\n这是一条测试消息，不代表实际人员事件或安全告警。\n通知方式已选择群消息，电话加急未启用。',signing_secret))
                save_credentials({'webhook':webhook,'secret':signing_secret,
                    'management_admin_token':secrets.token_urlsafe(32),'phone_enabled':False},GROUP_CREDENTIALS)
                config=json.loads((ROOT/'config'/'management.json').read_text(encoding='utf-8'))
                service=ManagementNotifications(ROOT/'runtime'/'management.sqlite3',config)
                try:
                    service.set_routes({'role':'admin','id':'local-group-setup'},[{
                        'id':'laboratory_group','transport':'webhook','labs':[config['lab_id']],
                        'event_types':['person.entered','instrument.started','safety.no_mask'],
                        'webhook_env':'LAB_FEISHU_GROUP_WEBHOOK','secret_env':'LAB_FEISHU_GROUP_SECRET'}])
                finally:
                    service.db.close()
                status('connected')
                root.after(0,finished,True,'飞书已接受群测试消息；凭证已加密保存，通知路由已切换为群。请在飞书群确认收到，随后启动通知服务。')
            except Exception as exc:
                detail=str(exc) if isinstance(exc,FeishuError) else '网络或本机保存失败（'+type(exc).__name__+'）。'
                status('failed',detail)
                # Exception strings may contain the secret-bearing webhook URL.
                root.after(0,finished,False,detail)
        threading.Thread(target=connect,daemon=True).start()
    button=ttk.Button(frame,text='连接并向此群发送测试消息',command=submit)
    button.pack(anchor='w',pady=14)
    status('awaiting_input')
    root.protocol('WM_DELETE_WINDOW',lambda:(status('closed'),root.destroy()))
    root.mainloop()


if __name__=='__main__':
    main()
