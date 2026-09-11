# 实验室安全摄像头

## 当前架构

当前部署采用“实验室主机主动推送 + lnx GPU 集群计算”：

```text
海康 192.168.31.191/190 + TP-LINK 192.168.31.192
        │ 实验室局域网 RTSP
实验室主机：FFmpeg 三路推流
        │ 127.0.0.1:18554，经 SSH 转发到 lnx:8554
        ▼
lnx：MediaMTX + YOLOv8n + ByteTrack + ArcFace + Re-ID + Flask
        ▲
观察端：SSH API 隧道 + 浏览器
```

实验室主机只负责读取三台摄像头并推流，不对外开放 RTSP 或 Flask 端口，也不依赖固定的 Tailscale 地址。ln01 只作为 SSH 登录和 Slurm 提交入口；媒体接收与推理服务运行在 lnx。

三路摄像头配置如下：

```text
cam_1        192.168.31.191  /Streaming/Channels/102
cam_2        192.168.31.190  /Streaming/Channels/102
cam_entrance 192.168.31.192  /stream1
```

## 启动 RTSP 链路

先在实验室主机从仓库根目录启动 SSH 隧道。若本机 `5000` 已被开发服务占用，可把集群 API 映射到 `15000`：

```powershell
powershell -ExecutionPolicy Bypass -File .\camera\scripts\start_cluster_tunnel.ps1 `
  -ApiLocalPort 15000
```

隧道使用当前 Windows 用户目录下的 `.ssh\lab_rtsp_ed25519`，把本机 `18554` 转发到 `lnx:8554`。

然后启动三路推流：

```powershell
powershell -ExecutionPolicy Bypass -File .\camera\scripts\start_rtsp_push.ps1 `
  -RemoteBaseUrl "rtsp://127.0.0.1:18554"
```

启动器会分别提示输入三台摄像头密码。密码只存在于当前进程内存，不写入代码、配置或日志。海康输入默认使用 TCP，输出固定使用 TCP，视频流直接复制，不重新编码。任一 FFmpeg 进程退出后，启动器会自动重启该路。

需要随 Windows 登录自动恢复时，以当前仓库位置注册计划任务：

```powershell
powershell -ExecutionPolicy Bypass -File .\camera\scripts\register_rtsp_startup.ps1
```

注册脚本从自身位置解析隧道和推流脚本，不依赖固定磁盘路径。

## 集群接收与预检

在 ln01 上提交长期 RTSP 接收作业：

```bash
cd ~/lab-safety-host
sbatch cluster/rtsp_receiver.slurm
```

接收作业运行在 lnx，监听 RTSP `8554` 和 HLS `8888`。集群内部使用以下地址：

```text
LAB_CAM_1_RTSP=rtsp://127.0.0.1:8554/cam_1
LAB_CAM_2_RTSP=rtsp://127.0.0.1:8554/cam_2
LAB_CAM_ENTRANCE_RTSP=rtsp://127.0.0.1:8554/cam_entrance
```

提交预检：

```bash
sbatch --partition=gpu --nodelist=lnx --gres=gpu:1 --time=00:10:00 \
  --output=lab-camera-check-%j.out --error=lab-camera-check-%j.err \
  cluster/test_lnx_camera.slurm
```

成功结果应包含：

```text
cam_1_hls=ok
cam_2_hls=ok
cam_entrance_hls=ok
RTSP push relay read test passed from lnx.
```

如果端口开放但提示 `no stream is available`，说明 MediaMTX 接收器仍在运行，但实验室主机的 SSH 隧道或 FFmpeg 推流没有运行。

## GPU 服务

预检成功后提交实时服务：

```bash
cd ~/lab-safety-host
export LAB_CAM_1_RTSP=rtsp://127.0.0.1:8554/cam_1
export LAB_CAM_2_RTSP=rtsp://127.0.0.1:8554/cam_2
export LAB_CAM_ENTRANCE_RTSP=rtsp://127.0.0.1:8554/cam_entrance
bash cluster/submit_camera_service.sh --start-service
```

GPU 服务运行 YOLOv8n、ByteTrack、ArcFace、Re-ID、地图投影、录像和告警。采集线程只保留最新帧；处理速度不足时丢弃旧帧，避免累计解码延迟。

## 打包和上传

从仓库根目录生成部署包：

```powershell
powershell -ExecutionPolicy Bypass -File .\cluster\make_bundle.ps1
scp .\camera_cluster_bundle.tar.gz xqyi@10.137.145.22:~/
```

ln01 解压：

```bash
mkdir -p ~/lab-safety-host
tar -xzf ~/camera_cluster_bundle.tar.gz -C ~/lab-safety-host
```

部署包不会包含摄像头密码、本地虚拟环境、运行缓存或模型输出。

## 观察页面

`start_cluster_tunnel.ps1` 默认同时映射集群 API。如果使用 `-ApiLocalPort 15000`，再启动本地静态观察页：

```powershell
powershell -ExecutionPolicy Bypass -File .\camera\scripts\start_observer.ps1 `
  -BackendPort 15000
```

浏览器打开 `http://127.0.0.1:5173/?api=http://127.0.0.1:15000&remote=1`。端口 `15000` 是仅供观察页调用的集群 API，不是前端页面。

也可以单独建立 API 隧道：

```powershell
ssh -N -i "$env:USERPROFILE\.ssh\lab_rtsp_ed25519" `
  -o IdentitiesOnly=yes `
  -L 15000:lnx:5000 xqyi@10.137.145.22
```

观察端不直接连接摄像头，也不运行推理模型。
