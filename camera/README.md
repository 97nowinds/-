# 实验室安全摄像头

## 当前架构

当前部署采用“实验室主机主动推送 + lnx GPU 集群计算”：

```text
海康 192.168.1.64/65 + TP-LINK 192.168.1.66
        │ 实验室内网 RTSP
实验室主机 100.126.39.4
        │ MediaMTX + FFmpeg -c:v copy，TCP 8554，仅 Tailscale
        ▼
lnx GPU 节点：YOLOv8n、ByteTrack、ArcFace、Re-ID、地图、录像、告警、Flask
        ▲
笔记本：SSH 隧道 + 浏览器，不连接摄像头，不运行模型
```

实验室主机只运行三个 FFmpeg 推送进程。它不运行 MediaMTX、Python、YOLO、ArcFace、Re-ID、地图算法、录像或 Flask。ln01 只用于 SSH 登录、文件上传和提交 Slurm 作业，不能作为计算节点或视频中转节点。

三路 RTSP 转发已经配置：两台海康使用 `/Streaming/Channels/102`，入口 TP-LINK 使用 `/stream1`。入口流先只转发和验证，启用 `LAB_CAM_ENTRANCE_RTSP` 后才由 GPU 作业执行人脸识别。

## 主机主动推送

主机不再需要开放入站 RTSP 端口，也不需要 Tailscale。管理员必须先提供一个真正的 RTSP ingest 地址；`10.137.145.22:22` 是 SSH，不是 RTSP 接收端，不能直接使用。

普通 PowerShell 启动主动推送：

```powershell
cd C:\codex1\camera
powershell -ExecutionPolicy Bypass -File .\scripts\start_rtsp_push.ps1 `
  -RemoteBaseUrl "rtsp://<INGEST_HOST>:<INGEST_PORT>/lab" `
  -RemoteUsername "<INGEST_USER>"
```

如果接收端不需要认证，省略 `-RemoteUsername`。启动时输入两台海康和入口 TP-LINK 的密码，以及（如需要）接收端密码。密码只在当前进程内存中使用，不写入代码、配置或日志。FFmpeg 从三台内网摄像头拉流，再主动推送为：

```text
rtsp://<INGEST_HOST>:<INGEST_PORT>/lab/cam_1
rtsp://<INGEST_HOST>:<INGEST_PORT>/lab/cam_2
rtsp://<INGEST_HOST>:<INGEST_PORT>/lab/cam_entrance
```

输出侧固定为 TCP；输入侧默认 TCP，可用 `-RtspTransport udp` 切换海康输入协议。推送使用 `-c:v copy -c:a copy`，不解码、不重新编码。FFmpeg 任一进程退出后，启动器自动重启该路。

主动推送模式的远端接收端必须由服务器管理员预先启动，并提供可写入的 RTSP 路径。主机端没有可供 lnx 反向读取的 `100.126.39.4:8554` 服务。

## lnx 预检

正式运行时先由 ln01 提交长期 RTSP 接收作业，不能依赖交互终端中的 `nohup`：

```bash
cd ~/lab-safety-host
sbatch cluster/rtsp_receiver.slurm
```

接收作业固定运行在 lnx，申请 1 CPU、1 GB 内存，不申请 GPU，并监听 RTSP `8554` 和 HLS `8888`。实验室主机的 SSH 隧道及主动推流脚本会在接收端重启后自动重连。

本阶段不要提交 `camera_service.slurm`。先把测试脚本单独上传到 ln01，再由 ln01 提交到 lnx：

```bash
sbatch --partition=gpu --nodelist=lnx --gres=gpu:1 --time=00:10:00 \
  --output=lab-camera-check-%j.out --error=lab-camera-check-%j.err \
  cluster/test_lnx_camera.slurm
```

也可以在下一阶段使用自动提交脚本，但它会同时提交预检和服务作业；当前阶段只执行上面的预检作业。查看结果：

```bash
squeue -u "$USER"
cat lab-camera-check-<作业号>.out
```

预检必须在输出中显示 lnx 主机名、TCP 端口成功，并且两条 FFmpeg 读流命令都成功：

```text
cam_1_frame=ok
cam_2_frame=ok
cam_entrance_frame=ok
RTSP relay read test passed from lnx.
```

TCP 端口成功只说明网络路径存在。真正的 RTSP 读流确认应在 lnx 使用 FFmpeg 读取两路各一帧：

```bash
ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
  -i rtsp://100.126.39.4:8554/cam_1 -map 0:v:0 -frames:v 1 -f null -
ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
  -i rtsp://100.126.39.4:8554/cam_2 -map 0:v:0 -frames:v 1 -f null -
```

若 lnx 的 `nc` 成功但 FFmpeg 首帧失败，检查 MediaMTX/FFmpeg 网关进程、海康密码和路径；若 `nc` 失败，需要学校网络管理员为 lnx 开通到实验室 Tailscale 地址的路由或防火墙策略。

## 集群作业（下一阶段）

集群文件位于 `C:\codex1\cluster`：

```text
camera_service.slurm       lnx GPU 实时服务作业
test_lnx_camera.slurm      GPU 节点和 RTSP 预检
submit_camera_service.sh   预检后提交服务的入口
check_lnx_api.sh           检查 Flask /api/health 和 /api/state
requirements.cluster.txt   GPU 版依赖，使用 onnxruntime-gpu
make_bundle.ps1             生成无密码部署包
```

部署包生成和上传命令：

```powershell
powershell -ExecutionPolicy Bypass -File C:\codex1\cluster\make_bundle.ps1
scp C:\codex1\camera_cluster_bundle.zip xqyi@10.137.145.22:~/
```

ln01 解压：

```bash
mkdir -p ~/lab-safety-host
unzip -o ~/camera_cluster_bundle.zip -d ~/lab-safety-host
```

确认两路 RTSP 读流成功后，才在 ln01 执行下一阶段的 GPU 作业：

```bash
cd ~/lab-safety-host
bash cluster/submit_camera_service.sh --start-service
```

作业申请 `gpu` 分区、`lnx` 节点、1 张 GPU，并把项目复制到 `$SLURM_TMPDIR` 本地盘。默认环境变量是：

```text
LAB_CAM_1_RTSP=rtsp://100.126.39.4:8554/cam_1
LAB_CAM_2_RTSP=rtsp://100.126.39.4:8554/cam_2
LAB_CAM_ENTRANCE_RTSP=rtsp://127.0.0.1:8554/cam_entrance
```

入口摄像头通过 `LAB_CAM_ENTRANCE_RTSP=rtsp://127.0.0.1:8554/cam_entrance` 启用；不设置时，双摄模式仍可独立运行。摄像头密码不进入集群环境变量，因为集群只读取实验室主机已经发布的无密码转发地址。

## 笔记本观察

GPU 服务作业在 lnx 上监听 `0.0.0.0:5000` 后，笔记本只建立 SSH 隧道：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 5000:lnx:5000 xqyi@10.137.145.22
```

然后浏览器打开：

```text
http://127.0.0.1:5000/
```

笔记本不会访问 `192.168.1.64`、`192.168.1.65` 或 `100.126.39.4:8554`，也不安装或运行推理依赖。

## 实时性与功能

lnx 服务沿用项目的最新帧策略：采集线程只保留最新帧，处理速度不足时丢弃旧帧，不累计解码队列延迟。两路摄像头运行 YOLOv8n、单摄 ByteTrack、Re-ID、地图投影、录像和告警；入口摄像头可选，用 `LAB_CAM_ENTRANCE_RTSP` 启用 SCRFD + ArcFace 身份确认及受方向、时间和相似度限制的交接。
