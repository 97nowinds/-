"""Local operator access; remote access retains maintenance authentication."""
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request,build_opener,ProxyHandler
from urllib.error import HTTPError


def operator(handler, runtime):
    if runtime is None:return None
    principal=runtime.service.authenticate(handler.headers.get('Authorization',''))
    if principal and principal['role']=='admin':return principal
    host=handler.headers.get('Host','')
    try:hostname=urlsplit('//'+host).hostname
    except ValueError:return None
    if (handler.client_address[0] not in {'127.0.0.1','::1'}
            or hostname not in {'127.0.0.1','localhost','::1'}
            or handler.headers.get('Sec-Fetch-Site') in {'cross-site','same-site'}):return None
    origin=handler.headers.get('Origin')
    if origin and origin!='http://'+host:return None
    return {'id':'local-face-calibration','role':'admin'}


def handle_face_calibration(handler, method):
    path=urlsplit(handler.path).path
    if path=='/face-calibration' and method=='GET':
        handler._send_file(Path(__file__).parent/'templates/face_calibration.html','text/html; charset=utf-8')
        return True
    prefix='/api/management/face-calibration/'
    if not path.startswith(prefix):return False
    runtime=getattr(handler.server,'management',None)
    principal=operator(handler,runtime)
    if not principal:
        handler._send_json({'error':'请在本机通过 localhost 或 127.0.0.1 打开此页面'},status=403);return True
    action=path[len(prefix):]
    allowed=(method=='GET' and (action=='status' or re.fullmatch(r'images/[a-f0-9]{32}\.jpg',action))) or (method=='POST' and action in {'start','stop','review','unreview','enroll','reload'})
    if not allowed:handler._send_json({'error':'not found'},status=404);return True
    from feishu_app import load_credentials
    try:
        token=load_credentials(Path(__file__).parent/'runtime/face-adaptation.dpapi')['token']
        data=None
        if method=='POST':
            length=int(handler.headers.get('Content-Length','0'))
            if not 0<length<=8192:raise ValueError('请求过大')
            data=handler.rfile.read(length)
            if not isinstance(json.loads(data),dict):raise ValueError('请求无效')
        req=Request('http://127.0.0.1:15003/'+action,data=data,method=method,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with build_opener(ProxyHandler({})).open(req,timeout=40) as response:
            body=response.read(2*1024*1024)
        if action.startswith('images/'):
            handler._send_bytes(body,'image/jpeg')
        else:
            payload=json.loads(body)
            handler._send_json(payload)
            if method=='POST':
                with runtime.service.lock,runtime.service.db:
                    runtime.service._audit(principal['id'],'face-calibration.'+action,'reviewed-site-samples')
    except HTTPError as error:
        try:message=json.loads(error.read(8192)).get('error','采集服务拒绝请求')
        except Exception:message='采集服务拒绝请求'
        handler._send_json({'error':str(message)[:200]},status=error.code)
    except Exception:
        handler._send_json({'error':'集群采集服务暂不可用'},status=503)
    return True
