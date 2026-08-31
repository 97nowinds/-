import os
from urllib.parse import urlsplit, urlunsplit


def classify_source(source):
    if isinstance(source, int):
        return "usb"
    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        return "rtsp"
    if isinstance(source, str):
        return "file_or_url"
    return "unconfigured"


def display_source(source, source_env=None):
    if source is None:
        return f"环境变量 {source_env} 未配置" if source_env else "未配置"
    if isinstance(source, int):
        return f"USB 摄像头 {source}"
    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        parsed = urlsplit(source)
        host = parsed.hostname or "已配置主机"
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))
    return str(source)


def resolve_camera_source(camera, fallback_index=0, environ=None):
    environ = os.environ if environ is None else environ
    source_env = camera.get("source_env")
    if source_env:
        source = environ.get(source_env, "").strip()
        if not source:
            return {
                "source": None,
                "source_env": source_env,
                "source_type": "unconfigured",
                "source_display": display_source(None, source_env),
                "configuration_error": f"未设置环境变量 {source_env}",
            }
    else:
        source = camera.get("source", fallback_index)

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    return {
        "source": source,
        "source_env": source_env,
        "source_type": classify_source(source),
        "source_display": display_source(source, source_env),
        "configuration_error": None,
    }
