"""Send a generated lab daily report to a configured Feishu application-bot chat."""

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


CAMERA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMERA))
from feishu_app import BASE, EVIDENCE_CREDENTIALS, FeishuAppClient, FeishuError, load_credentials


def upload_report(client, report):
    contents = report.read_bytes()
    if not contents:
        raise FeishuError("日报为空")

    boundary = uuid.uuid4().hex
    header = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file_type"\r\n\r\nstream\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file_name"\r\n\r\n{report.name}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{report.name}"\r\n'
        'Content-Type: application/octet-stream\r\n\r\n'
    )
    body = header.encode() + contents + f'\r\n--{boundary}--\r\n'.encode()
    request = Request(BASE + '/im/v1/files', data=body, method='POST', headers={
        'Authorization': 'Bearer ' + client.token(),
        'Content-Type': 'multipart/form-data; boundary=' + boundary,
    })
    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read(65536))
    except HTTPError as error:
        try:
            result = json.loads(error.read(65536))
        except ValueError:
            raise FeishuError(f"日报上传失败，HTTP {error.code}") from None
    except (OSError, ValueError):
        raise FeishuError("日报上传连接失败或响应无效") from None

    key = result.get('data', {}).get('file_key')
    if result.get('code') != 0 or not key:
        raise FeishuError(f"日报上传失败，飞书错误码 {result.get('code')}")
    return key


def send_once(client, chat_id, report_name, kind, content, receipt_dir):
    delivery_id = uuid.uuid5(uuid.NAMESPACE_URL, f"aims-feishu-daily/{chat_id}/{report_name}/{kind}").hex
    receipt = receipt_dir / f"{delivery_id}.sent"
    if receipt.is_file():
        return False

    receipt_dir.mkdir(parents=True, exist_ok=True)
    result = client.request('POST', '/im/v1/messages?receive_id_type=chat_id', {
        'receive_id': chat_id,
        'msg_type': kind,
        'content': json.dumps(content(), ensure_ascii=False),
        'uuid': delivery_id,
    }, client.token())
    message_id = result.get('data', {}).get('message_id')
    if not message_id:
        raise FeishuError("飞书未返回消息 ID，无法确认日报送达")
    receipt.write_text(message_id, encoding='utf-8')
    return True


def send_report(client, chat_id, report, receipt_dir, summary=None):
    sent = send_once(client, chat_id, report.name, 'file',
                     lambda: {'file_key': upload_report(client, report)}, receipt_dir)
    if summary:
        sent_summary = send_once(client, chat_id, report.name, 'text',
                                 lambda: {'text': summary}, receipt_dir)
        sent = sent or sent_summary
    return "飞书日报已发送" if sent else "当天的飞书日报已经发送"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True, type=date.fromisoformat)
    parser.add_argument('--chat-id', required=True)
    parser.add_argument('--summary', help='Optional group message with the daily work and push results')
    args = parser.parse_args()
    if not re.fullmatch(r'oc_[a-zA-Z0-9]+', args.chat_id):
        parser.error("群 ID 必须是 oc_ 开头的飞书 chat_id")

    report = CAMERA.parent / 'docs' / 'daily' / f'{args.date.isoformat()}.md'
    if not report.is_file():
        parser.exit(1, f"日报文件不存在：{report.name}\n")
    try:
        client = FeishuAppClient(load_credentials(EVIDENCE_CREDENTIALS))
        print(send_report(client, args.chat_id, report, CAMERA / 'runtime' / 'feishu-daily-receipts', args.summary))
    except (FeishuError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == '__main__':
    main()
