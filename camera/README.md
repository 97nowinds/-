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

## MTMC 2.0 数据流

```text
RTSP host_receive 时间戳
  -> YOLOv8n 人员检测
  -> 每路独立 ByteTrack 本地轨迹
  -> 质量门控的 ResNet50-IBN 轨迹 gallery
  -> 短窗口候选收集与硬约束过滤
  -> 匈牙利一对一关联
  -> high 继承 / medium pending / low 匿名
  -> 全局 ID 事件、平面图与兼容 API

cam_entrance ArcFace 注册身份
  -> 身份单所有者锁
  -> 已允许方向的跨摄交接
  -> 迟到证据 merge/redirect
```

算法参数集中在 `config/mtmc.json`。启动时会校验阈值顺序、范围、权重和事件路径；最终生效值及 `mtmc-2.0.0` 算法版本可从 `/api/state` 的 `processing.effective_mtmc_config` 查看。入口语义来自 `config/cameras.json` 的 `role`，通用关联代码不硬编码入口摄像头 ID。

## 标定状态

`config/floor_map.json` 使用三态 `calibration.status`：

- `uncalibrated`：没有可用的统一地面映射。只使用人员类别、摄像头有向拓扑、时间窗和 ReID。
- `approximate`：允许显示近似位置，但地图距离和重叠区不能成为高置信度合并依据。
- `formal`：现场控制点验收完成后，才允许统一地面坐标的严格距离门控和双摄位置融合。

当前配置为 `approximate`：三路摄像头都使用操作员选取的近似单应性；SLAM 为 relative 且没有 ArUco marker。因此当前地图只能用于操作提示，不能证明两个人是同一人。入口摄像头不再给多人复制固定锚点。每路状态的 `localization` 字段公开定位模式、标定状态、几何融合许可、最近质量和降级原因；明显流间偏差也会关闭几何融合。

近似标定下，同一个 global ID 在重叠画面中会保持既有地图坐标来源，避免多台摄像头逐帧争抢造成跳点；真正换源时通过卡尔曼滤波和 `calibration.max_map_speed_mps` 人体速度门控渐进过渡。投影落入实验台、柜体或通风设备实体范围时会移到最近可行走边界。地图人物状态公开 `position_source_changed`、`motion_limited`、`raw_position_step_m` 和 `walkability_adjusted`，用于区分正常行走、跨摄换源和近似标定修正。这只能稳定示意轨迹，不能替代现场正式控制点标定。

当前现场布局还通过 `floor_map.json` 的 `tracking_allowed_zone_ids` 将人物坐标锁定在副通道、主通道和侧向通道的并集。中央长实验台、沿墙柜台和通风设备区仍保留在底图中，但不会作为人物位置；落在这些区域的近似测量会投影到最近的允许通道。跨摄重叠区仍参与关联判断，其人物位置在状态和界面中按所属主通道或侧向通道显示。

跨通道移动按这些矩形区域的真实连通关系路由。副通道和主通道不直接相邻，因此两者之间的坐标迁移必须依次经过副通道与侧向通道的交口、侧向通道、侧向通道与主通道的交口；不会再用穿过中央实验区的欧氏直线后突然吸附到另一条通道。人物状态中的 `corridor_routed` 可用于确认本次更新是否启用了连通路径约束。测量中断后恢复时，`calibration.max_motion_gap_seconds` 限制单次位置更新可以使用的时间跨度，避免保持点在切换摄像头后瞬移到新观测位置。

单应性只在四个图像控制点围成的区域内可信。`projection_domain_margin_normalized` 为检测框脚点预留少量边缘抖动容差；超出该范围的点不会继续外推并吸附到另一条通道，而是停止发布该路原始地图观察。最近一次判断通过 `localization.recent_projection` 暴露，域外点会给出 `footpoint outside calibrated image domain`，避免把“没有可靠坐标”伪装成一条平滑但错误的轨迹。

摄像头标注页中的主通道、副通道和侧向通道现在也是实时跟踪 ROI，而不再只是可视化参考。YOLO/ByteTrack 目标的脚点必须位于标注通道内；存在四点单应性时还必须位于四点围成的有效域内。通道外目标不会进入身份、跨摄关联或地图状态，`roi_rejected_tracks` 会报告最近一轮被排除的数量。重叠区只描述交接证据，不会单独扩大允许跟踪范围；没有通道标注和四点映射的旧配置保持不限制，以兼容入口摄像头。

通道区域同时支持旧式矩形和四点透视多边形。cam_1 的副通道使用与其地面参考点一致的凸四边形，因此标注层不会把左上方实验台误画成可跟踪通道；主通道仍保持平行四边形约束，避免编辑时产生畸形区域。

若 YOLO/ByteTrack 和 global ID 仍持续存在，仅仅是地图投影暂时无效，则地图最多按 `calibration.map_position_hold_seconds` 保留最后一个可信点（当前依据录像中侧向通道到主通道约 7.4 秒的有效域缺口设置为 8 秒），并返回 `position_estimated: true`、`localization_state: held` 和坐标年龄。前端以虚线光环和“定位保持”标识这段不确定状态，重新取得有效投影后沿同一轨迹继续；它不会虚构盲区内的运动。真正的跟踪消失或保持超时后才断开轨迹。

集群作业必须设置 `LAB_FLOOR_MAP_PATH` 指向共享项目目录中的 `camera/config/floor_map.json`。Slurm 脚本已默认这样设置，所以标注保存不会再落到 `/cluster/dataX/.../lab-camera-<job>` 临时副本。标注 API 同时使用配置修订号防止多个旧页面互相覆盖；普通区域编辑明确不会改写摄像头四点，只有“地面参考点”发生编辑时才更新单应性控制点。遇到 HTTP 409 时刷新标注页后重新操作，不要覆盖较新的现场配置。

正式标定时，应为三路摄像头各采集至少四个不共线地面控制点，填写归一化图像点和同一实验室坐标系的米制地图点，检查重投影误差，再把状态改为 `formal`。若使用 SLAM 锚定，还必须给 `slam.markers` 填入现场 ArUco ID 与对应 `map_point`，不能用虚构坐标。

## 轨迹级 ReID gallery

每条 `(camera_id, local_track_id)` 维护有界 gallery。样本包含 2048 维 L2 归一化特征、帧时间、图像质量、质量分项和摄像头 ID；原始向量不会进入状态 API、事件日志或部署包。质量综合人体框面积、边界裁剪、清晰度、亮度和人体长宽比，低于阈值的帧被拒绝。近重复视角会去重，超容量时按质量与视角差异共同淘汰。跨轨迹比较使用质量加权 top-k 相似度；`appearance` 单向量仍保留作旧接口和空 gallery 回退。

状态中的每条 `yolo_tracks` 会公开 `reid_gallery.gallery_size`、有效/拒绝/重复样本数、最后更新时间、关联置信度和最终分数，但不返回 embedding。

## 批量关联与置信度

每次 YOLO 推理结果作为一个批次处理。关联器先检查 person 类别、有向摄像头拓扑、最大通行时间、过期候选和同时出现冲突，再融合 gallery ReID、时间差、正式标定时的地图距离和可用运动方向。三摄像头规模使用确定性的匈牙利算法完成一对一分配，输入列表或线程先后不改变同一批次的结果。

在当前 `approximate` 标定状态下，所有摄像头转换都关闭 `simultaneous_overlap_validated`：两个视角同时看到人员时不会仅凭相似实验服自动继承 global ID，而是暂时保持独立匿名轨迹，待来源视角消失后再用一对一证据完成交接。协调器的最终 merge 还会拒绝任何导致同一摄像头两个活跃 local ID 共用一个 global ID 的操作；地图层会防御性拆分上游重复 ID，避免两个人被绘制成一个坐标点。正式标定并完成双人重叠验收后，才应逐条重新启用同时重叠融合。

- `high`：分数、ReID 和唯一性 margin 均达标，允许继承已有 global ID。
- `medium`：建立临时匿名 ID 并进入 pending，等待后续 gallery 证据；不会强行显示为已知人员。
- `low`：创建临时匿名 ID。

不同已知人员永不自动合并，已注册身份继续使用单所有者锁。迟到高置信度证据可以产生 `merge` 和 `redirect`；redirect 表为未来人工 split/纠错保留了明确边界。

## 时间戳与诊断

OpenCV RTSP 当前无法可靠获得设备原始曝光时间，因此时间戳来源明确标记为 `host_receive`，不是 `camera_capture`。捕获线程记录 Unix 时间和单调时间；同一帧时间贯穿处理、SLAM、地图观察与跨摄关联。TTL、窗口和帧龄使用单调时钟，对外展示使用 Unix 时间。

`/api/state` 每路摄像头包含捕获接收时间、推理开始/结束时间、状态发布时间、帧龄、推理耗时和估计流间偏差。流间偏差超过 `calibration.max_stream_skew_seconds` 时不允许高置信度空间融合，并在 `localization.degraded_reason` 说明原因。

最近跨摄候选及结果可查询：

```text
GET /api/mtmc/events?limit=50
```

每个事件记录源/目标摄像头、本地轨迹、global ID、ReID/时间/几何/融合分数、采用或拒绝原因、置信度和算法版本。事件以追加 JSONL 保存在 `runtime/mtmc_events.jsonl`，默认保留 30 天并只恢复历史查询；服务重启绝不恢复已过期的实时轨迹。日志会过滤 embedding 等生物特征字段，也不写 RTSP 完整凭据。

## 录制、标注与离线评测

先从页面或 `/api/recording/start` 使用 `DatasetRecorder` 同步录制三路原始视频和 `*_timestamps.csv`。复制 `config/ground_truth.example.json` 后人工填写：

### 完整路线人工交接标记

观察页顶部的“开始录像”适合一次走完完整路线。输入匿名实验编号（例如 `worker_001`），开始录像后按计划依次经过入口、`cam_1` 和 `cam_2`，走出最后一路画面后停止录像。不要把姓名写入编号或备注。

停止后点击顶部“路线复核”，也可以直接打开：

```text
http://127.0.0.1:5173/recordings?api=http://127.0.0.1:15000
```

在复核页选择录像会话，然后对每一次交接执行：

1. 选择源摄像头和目标摄像头。
2. 拖动源视频到人员最后仍可见的画面，逐帧微调后点击“使用源视频当前画面”。
3. 拖动目标视频到人员首次可见的画面，点击“使用目标视频当前画面”。
4. 填写同一个匿名真实人员 ID 并保存。
5. 入口到 `cam_1`、入口到 `cam_2`、`cam_1` 到 `cam_2` 等交接分别保存；反向路线要另录或另标。

后端先把播放器时间按该路录像 FPS 映射到准确帧号，再从对应 `*_timestamps.csv` 取得 `host_receive` Unix 时间，因此不同摄像头的启动偏差和采集间隔不会被误当成同一视频时间轴。结果保存在会话目录的 `manual_handoffs.json`，包含源末帧、目标首帧、两路 Unix 时间和真实 gap。页面按摄像头方向显示样本数、中位数和 P90；这些只是人工标定建议，单次样本不会自动修改线上 `max_transit_seconds`。建议每条方向至少采集 5 次正常步速，并另采慢速、遮挡和交叉场景后再调整配置。

复核页只列出已经停止且状态为 `complete` 的录像；正在写入的视频不会开放复核。删除某条人工标记只删除 `manual_handoffs.json` 中对应记录，不删除原始录像。

- `camera_id`；
- `frame_index` 或 `unix_time`；
- `local_detection.local_id` 与可选 `box`；
- 真实 `person_id`；
- `attributes` 中的 `occluded`、`registered`、`crowd_size`；
- `handoffs` 中的人员、源/目标摄像头和交接时间区间。

预测文件格式参考 `config/predictions.example.json`。运行：

```powershell
.\.venv\Scripts\python.exe .\mtmc_evaluate.py `
  --session .\data\recordings\<subject>\<session> `
  --ground-truth .\data\recordings\<subject>\<session>\ground_truth.json `
  --predictions .\data\recordings\<subject>\<session>\predictions.json `
  --output .\data\recordings\<subject>\<session>\mtmc_report
```

`--session` 会读取 `info.json` 和各路 `*_timestamps.csv`，校验标注帧确实存在，并把 recorder 的 host_receive 时间补入只有帧号的标注。输出机器可读 JSON 和人可读 Markdown，包含跨摄交接成功率、错误身份继承率、漏交接率、ID switches、IDF1，以及按摄像头对、人数、遮挡和注册状态分组的结果。本工具没有把自定义指标称为 HOTA；当前未固定并接入标准 HOTA 依赖。空标注会明确输出“尚无真实评测数据”，不会生成虚假成绩。

## 当前已知限制

- 当前地图不是正式标定；三路虽都有逐像素近似映射，但四点有效域以外会主动停止发布坐标，几何仅供显示。
- ArUco marker 列表为空，relative SLAM 只能从近似单应性建立参考，不能提升到正式标定。
- `host_receive` 能衡量主机接收偏差，不能替代摄像头硬件同步或设备时间戳。
- 相似实验服、长时间完全遮挡和跨摄盲区仍可能使目标保持匿名或产生 ID switch；策略优先避免错误认人。
- 事件只持久化关联历史，不持久化实时 tracker 状态；服务重启后会创建新的实时轨迹。
- 尚无现场人工 ground truth，因此不能声称真实准确率已经提升。

## 现场验收清单

按顺序录像、标注并运行评测，每项同时检查画面、地图、`/api/state` 和 `/api/mtmc/events`：

1. 单人从入口到 `cam_1`。
2. 单人从入口到 `cam_2`。
3. `cam_1` 与 `cam_2` 双向交接。
4. 两人同时进入。
5. 两人交叉。
6. 两人穿相似实验服。
7. 短时遮挡后恢复。
8. 长时离开后重新进入。
9. 摄像头断流并自动重连。
10. 服务重启，确认历史事件可查但实时轨迹不恢复。
11. 未注册人员始终保持匿名。
12. 人脸不可见时只依赖保守 ReID，不继承已知身份。
13. 将地图置为 `uncalibrated` 或使标定失效，确认几何融合关闭。
14. 人为制造明显 RTSP 延迟不一致，确认流间偏差告警且几何融合降级。
