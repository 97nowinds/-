# 交互识别远程部署

交互识别模块和模型只部署在 GPU 服务器。本机只运行摄像头采集、人员跟踪和网页 API，并通过 HTTP 推送 `cam_2` 的 JPEG 帧与人员框。

## 服务器

将 `camera/plugins/instrument_interaction` 和 `camera/interaction_server.py` 部署到 GPU 服务器。在服务器的 `camera` 目录执行：

```powershell
.\scripts\setup_instrument_interaction.ps1 -Python311Exe C:\Path\to\python311.exe
.\scripts\start_interaction_server.ps1 -ListenAddress 0.0.0.0 -Port 6000
```

服务器必须满足随附说明的 Python、CUDA、GPU 和模型要求。先用浏览器或 curl 确认 `http://服务器IP:6000/health` 返回 200。

## 本机

在本机运行普通摄像头服务启动脚本时传入服务器地址：

```powershell
.\scripts\start_lab_host.ps1 -InteractionServerUrl http://服务器IP:6000
```

如果服务器配置了令牌，在本机进程启动前设置同一个环境变量：

```powershell
$env:LAB_INTERACTION_TOKEN = "同一个令牌"
```

本机不需要安装交互模块的 Torch、MiniCPM、YOLO 或模型权重。`LAB_INTERACTION_SERVER_URL` 未配置时，启动脚本会阻止误启动；临时关闭交互功能可使用 `-DisableInstrumentInteraction`。

## 接口

- `GET /health`：服务器模型健康状态。
- `POST /api/interaction/frame`：multipart 上传 `frame`，以及 `camera_id`、`interaction_camera_id`、`timestamp`、`persons`、`after_event_sequence` 等字段。
- `GET /api/interaction/status`：服务器当前流会话。

服务端为每个摄像头只创建一个 `create_stream()` 会话，并持续调用 `process_frame()`；本机网络异常时只丢弃队列中的旧帧，不把异常伪装成“无交互”。
