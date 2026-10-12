"""Local masked setup UI; sends only the explicitly requested connection test."""
import json
from pathlib import Path
import secrets
import sys
import threading
import tkinter as tk
from tkinter import ttk
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feishu_app import FeishuAppClient, FeishuError, save_credentials
from management_notifications import ManagementNotifications

STATUS = ROOT / 'runtime' / 'feishu-setup-status.json'


def status(state, code=None):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({'state': state, 'detail': code}, ensure_ascii=False), encoding='utf-8')


def main():
    root = tk.Tk()
    root.title('飞书点对点通知接入 — 本机安全输入')
    root.geometry('680x465')
    root.attributes('-topmost', True)
    box = ttk.Frame(root, padding=22)
    box.pack(fill='both', expand=True)
    ttk.Label(box, text='飞书应用机器人接入', font=('Microsoft YaHei', 17)).pack(anchor='w')
    ttk.Label(box, text='填写已发布的企业自建应用。密钥加密保存在当前 Windows 用户下。').pack(anchor='w', pady=10)
    fields = {}
    for name, label, hidden in [('app_id', 'App ID（cli_开头）', False),
                                ('app_secret', 'App Secret', True),
                                ('recipient', '测试接收人：手机号 / 邮箱 / open_id', True)]:
        ttk.Label(box, text=label).pack(anchor='w', pady=(8, 2))
        entry = ttk.Entry(box, width=75, show='●' if hidden else '')
        entry.pack(fill='x')
        fields[name] = entry
    ttk.Label(box, text='点击下方按钮将向所填接收人发送一条“连接测试”私聊消息。\n本次不会发起电话加急，也不会发送人员录像或识别数据。').pack(anchor='w', pady=14)
    info = tk.StringVar(value='等待输入。请先确认该接收人已加入应用可用范围。')
    ttk.Label(box, textvariable=info, wraplength=620).pack(anchor='w')

    def complete(success, message):
        info.set(message)
        button['state'] = 'normal'
        if success:
            fields['app_secret'].delete(0, 'end')
            fields['recipient'].delete(0, 'end')

    def submit():
        values = {key: entry.get().strip() for key, entry in fields.items()}
        if not all(values.values()):
            info.set('请填写全部三项。')
            return
        button['state'] = 'disabled'
        info.set('正在验证应用、查询接收人并发送连接测试……')
        status('connecting')
        def connect():
            try:
                client = FeishuAppClient(values)
                recipient = client.resolve_recipient(values['recipient'])
                # Same UUID remains stable within this one connection attempt.
                client.send_text(recipient,
                    '实验室通知系统：飞书点对点连接测试。\n这是一条测试消息，不代表人员进入、仪器操作或安全告警。\n电话加急未启用。',
                    'connection.test:' + uuid.uuid4().hex)
                credentials = {'app_id': values['app_id'], 'app_secret': values['app_secret'],
                               'primary_open_id': recipient, 'phone_enabled': False,
                               'management_admin_token': secrets.token_urlsafe(32)}
                save_credentials(credentials)
                config = json.loads((ROOT/'config'/'management.json').read_text(encoding='utf-8'))
                service = ManagementNotifications(ROOT/'runtime'/'management.sqlite3', config)
                service.set_routes({'role': 'admin', 'id': 'local-feishu-setup'}, [{
                    'id':'primary_recipient', 'transport':'app', 'recipient_key':'primary',
                    'labs':[config['lab_id']], 'event_types':['person.entered','instrument.started','safety.no_mask']}])
                service.db.close()
                status('connected')
                root.after(0, complete, True, '飞书已确认接受测试消息，接入信息已加密保存。请在手机或电脑飞书中检查私聊；电话加急仍关闭。')
            except Exception as exc:
                message = str(exc) if isinstance(exc, FeishuError) else '本机保存或配置失败，请重新检查。'
                status('failed', message)
                root.after(0, complete, False, message)
        threading.Thread(target=connect, daemon=True).start()
    button = ttk.Button(box, text='连接并向上述接收人发送测试消息', command=submit)
    button.pack(anchor='w', pady=15)
    status('awaiting_input')
    root.protocol('WM_DELETE_WINDOW', lambda: (status('closed'), root.destroy()))
    root.mainloop()


if __name__ == '__main__':
    main()
