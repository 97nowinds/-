# 原始高清同帧 snapshot.frame 字段说明

日期：2026-10-11。生产者协议标签保持 `lab-handoff-1.2`。
这是人员定位侧的字段说明，接收方须核对兼容，不能只改成 1.1 标签或关闭严格校验。

## 样例取得

人员定位服务健康检查：`GET /api/health`，应返回
`service_revision=hd-same-frame-20261011-r1`、`freeze_enabled=true`。
经现有图片交接服务认证读取 `GET /api/handoff/frozen/cam_2`。
此接口先发布固定 JPEG，再返回该帧绑定的检测和地图观测。
图片为 `GET /api/handoff/snapshots/{snapshot_id}.jpg`，使用已有图片读取 Bearer 凭据。
凭据不放入 URL、脱敏请求或文档。

导出命令（仓库根目录）：

```powershell
$env:PYTHONPATH = "$PWD/camera;$PWD/camera/runtime/joint-demo-deps"
& camera/.venv/Scripts/python.exe camera/handoff_sample.py `
  --credentials-file camera/runtime/handoff-credentials.json `
  --output camera/runtime/handoff-samples/cam_2 `
  --archive camera/runtime/handoff-samples/archive
```

输出 `snapshot.jpg`、`snapshot.frame.json` 和 `verification.json`。
样例检查原始尺寸、SHA-256、8 MiB 大小上限、图片有效期以及真实人员和新鲜地图观测。
空场景、断流、历史不足或无有效地图位置会失败，不伪造人员或地图点。
普通连续快照保留 60 秒。人工选中的样例通过同一图片服务、同一认证和地址保留 24 小时，
服务需使用 `--sample-archive camera/runtime/handoff-samples/archive` 启动。
具体到期时间以请求 `image_expires_at` 为准；本地文件可继续用于离线核对。

## 字段

| 字段 | 类型与含义 |
| --- | --- |
| `schema_version` | 固定 `lab-handoff-1.2` |
| `event_type` | 固定 `snapshot.frame` |
| `event_id` | `snapshot.frame:` 加快照编号；同一快照请求重试保持内容不变 |
| `snapshot_id` | 捕获会话、摄像头和成功检测帧序号构成的唯一编号 |
| `captured_at` | 原始帧解码接收时间，带时区的 ISO 8601；不是发送或分析结束时间 |
| `occurred_at` | 快照封装时刻，和 `captured_at` 同一时间域 |
| `map_id/map_revision` | 当前地图标识与实际布局版本 |
| `zone_revision` | 现有操作区域约定；尚未确认时为 null，不能伪造已验收版本 |
| `camera.camera_id/source_frame_id` | 实际摄像头和这张源帧的唯一标识 |
| `camera.frame_width/frame_height` | JPEG 解码后的真实宽高，等于原始源帧宽高 |
| `camera.image_url` | 固定 JPEG 地址；不含密码，内容不会被新帧覆盖 |
| `camera.image_sha256` | JPEG 字节的 SHA-256 十六进制值 |
| `camera.image_available_at/image_expires_at` | 图片发布时间和到期时间；过期读取明确失败 |
| `camera.geometry.binding` | `detection_frame`，人员框与 JPEG 来自同一源帧 |
| `camera.geometry.transform` | 高清入口为 `identity`，无需接收方再次放大人员框 |
| `camera.geometry.detection_width/detection_height` | 对外人员框的坐标画幅，即高清 JPEG 宽高；内部检测可在缩小副本运行 |
| `camera.geometry.hd_width/hd_height` | 原始高清画幅宽高 |
| `camera.geometry.hd_alignment_status` | 手工样例为 `pending`；不能假称接收方已验收 |
| `camera.tracks[].track_id/global_id` | 同一稳定全局轨迹 ID，不按姓名或数组顺序重编号；服务重启会开始新轨迹会话 |
| `camera.tracks[].local_track_id` | 当前摄像头内部局部轨迹编号 |
| `camera.tracks[].person_id` | 已锁定注册人员 ID；未知身份为 null，禁止猜测 |
| `camera.tracks[].display_name` | 脱敏样例为 null；稳定 ID 保留 |
| `camera.tracks[].bbox/bbox_format` | `[x,y,宽,高]`，`xywh_pixels`，坐标属于高清 JPEG |
| `camera.tracks[].confidence` | 人员检测置信度；不当成地图定位质量 |
| `camera.tracks[].observed_at` | 当前帧人员检测对应的原始采集时刻 |
| `camera.tracks[].location` | 同一轨迹的地图观测；缺失时不可用于地图候选 |
| `location.x_m/y_m` | 米制地图坐标，原点和轴方向沿用地图接口 |
| `location.quality/trajectory_status/valid` | 实际定位质量、轨迹状态和观测有效性；近似标定不升级为正式精度 |
| `location.observed_at` | 该地图观测的真实时间，不覆盖成图像时间；与图像差值需通过现有 1 秒检查 |

## 时间域与验收边界

手工样例保留源主机时间：`camera.timing.event_time_domain=source`、
`conversion_applied=false`、`sync_status=unavailable`，
`capture_time_kind=decoder_received_at`。`source_clock` 区分采集会话。
此时用于协议、画质与坐标核对，不代表自动分析条件已经满足。

既有自动发送路径仍要求专用时钟同步、操作区版本、接收 Schema 和人工验收。
通过后才使用 authority 时间域，保留源时间、同步编号、偏移量、有效期和误差范围。
不能把源时间标成 authority，也不能通过放宽校验强行启动自动模式。

冻结入口只接受成功检测的当前源帧，跳过缓存框；摄像头断流、快照超过时效返回 503。
取图失败、过期、历史不足、身份歧义均表示不可用，不能表示有效的“没有交互”。
电子天平约 3 帧/秒的性能测试须在合格样例和接收兼容核对通过后进行。
