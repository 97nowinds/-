# 视觉与硬件火焰报警联合接入

日期：2026-10-15。状态：接入与融合代码、隔离测试完成；尚未重启生产服务，也未配置未经确认的覆盖区域。未新增或训练火焰视觉模型，需要实际视觉检测模块按以下契约提供结果。

## 10 月 15 日后续：已实现主项目视觉推理

新增 fire_detector.py、fire_worker.py 和 scripts/evaluate_fire_detector.py：主项目自行进行 ONNX 推理，不依赖队友模型。CPU 线程限制 2，使用独立进程按最新快照采样，无堆积队列。原始图像经过等比例缩放和 letterbox，输出框去除黑边并还原到该原始 JPEG 像素；fire 与 smoke 分开处理，烟雾不冒充火焰。

当前模型仍处于 experimental_observe，提交群通知被禁止。没有火焰框时只报告 unknown，不使用 1-检测分数伪造“无火焰置信度”，不自动发送 clear。现有恢复入口仍需要经过验证的清除判断；该候选模型自身尚不支持自动解除。

模型与实测详见 [视觉检测实现与评测](FIRE_DETECTOR_EVALUATION_20261015.md)。本节更新前文“未新增模型”的历史状态，正式通知接入和现场准确率仍未验收。

## 用户确认的安装点

用户提供的地图截图红框位于通风实验台靠主通道一侧、中间偏左。按现有 SVG 显示比例估算，图上参考坐标约 x=5.0、y=5.9（当前 12×7 的示意地图坐标）；这不是实测位置精度，也不是传感器半径。

原图：C:/Users/22606/AppData/Local/Temp/codex-clipboard-d4b664e0-c27c-4c3a-abf2-3ff16e19f861.png。截图红框用于说明安装点，不作为覆盖区多边形。示意视场中 Cam2 是待核对候选，但未据此启用整幅画面的双源合并。地图本身仍为 approximate，原画面的对应区域和硬件检测范围需单独核对。

## 行为

| 输入 | 通知与状态 |
| --- | --- |
| 本地硬件新鲜火焰报警 | 立即发 L3 到三个群，不等待视觉 |
| 视觉连续确认火焰 | 默认至少 2 个不同源帧、跨度至少 1 秒、置信度至少 0.9；发 L3 到三个群并附实际快照 |
| 同一区域两类来源均为新鲜报警，观测间隔不超过 10 秒 | 同一 incident_id 升级为 dual_confirmed；三个群收到一次双源确认通知 |
| 重复同帧、持续相同报警 | 更新状态，不不断生成重复群消息 |
| 只有一个来源恢复、另一个仍报警或 unknown | 保持事件活动，不解除 |
| 所有曾参与报警的来源均提供有效、新鲜的恢复信号 | 登记 recovered；仍需维护人员现场核查后关闭 |

视觉的 clear 也要求连续帧确认。unknown、低置信度、断线、过期不等于 clear；明确 unknown 会使先前 clear 失效，防止被另一路信号误解除。硬件观测新鲜度为 5 秒，视觉为 15 秒。视觉恢复过期或硬件失联时不自动清除既有火焰事件。

双源确认表示两类检测信号在已确认覆盖区域内共同报警，不能代替现场火情核查。默认阈值是可调的工程参数，尚未经过本实验室真实火焰与干扰画面的准确率验收。

## 区域映射必须先确认

当前 config/management.json 的 fire_fusion.camera_zones 与 device_zones 均为空，避免猜测传感器覆盖范围。未映射时：保留原本硬件独立报警，视觉按各 camera_id 独立报警，不自动合并。

只有两者实际监视同一物理区域，才应配置到同一个 zone_id。当前每路摄像头只能配置一个覆盖区域；如果一个视角包含多个传感器区域，不能直接把整幅画面映射到其中一个局部区域，需要后续增加火焰框与 ROI 的匹配后再融合。本轮没有重新绘制地图或 ROI。

以下仅为映射结构示例，**不是已验证的现场配置**：

```json
{
  "fire_fusion": {
    "camera_zones": {"cam_2": "synthetic_fire_zone"},
    "device_zones": {"synthetic_sensor": "synthetic_fire_zone"},
    "confirmation_window_seconds": 10,
    "visual_minimum_confidence": 0.9,
    "visual_minimum_frames": 2,
    "visual_minimum_seconds": 1,
    "visual_maximum_gap_seconds": 5
  }
}
```

hardware device_id 使用实际串口上报的设备编号。不能根据摄像头名称或地图点距离猜测覆盖区域，也不接纳视觉回传方自行指定的 zone_id。

## 视觉结果入口

POST /api/management/evidence，继续使用既有 producer Bearer Token，检查实验室与 camera_id 范围。新增 evidence_type=fire，不修改原有 snapshot.frame 传输格式。视觉结果可以来自队友或其他可信检测服务，不要求 person_id 或 track_id。

以下为**合成数据**，不能当作真实报警提交：

```json
{
  "evidence_type": "fire",
  "analysis_id": "synthetic-fire-analysis-002",
  "camera_id": "cam_2",
  "snapshot_id": "synthetic-frozen-frame-002",
  "time_domain": "authority",
  "captured_at": "2026-10-15T04:00:01.000+00:00",
  "published_at": "2026-10-15T04:00:02.000+00:00",
  "expires_at": "2026-10-15T04:00:07.000+00:00",
  "status": "valid",
  "fire_state": "detected",
  "confidence": 0.97,
  "fire_bbox": [100, 80, 120, 180],
  "bbox_format": "xywh_pixels"
}
```

- fire_state：detected / clear / unknown；布尔 is_interacting 或人员身份不是火焰结果。
- captured_at 必须等于原始快照采集/接收时间，不能填写结果发送时间。原快照自身必须标记 authority，不能仅由接收方声称已同步。
- expires_at 必须未过期，同时最多接纳到 published_at + 5 秒；采集观测最大年龄 15 秒。
- camera_id + snapshot_id 必须能在主项目的原始/共享快照索引中解析。回传 image_url、姓名等不能覆盖主项目记录。
- fire_bbox 可省略，省略时共享整张场景快照，不画虚假的火焰框；提供时必须在该快照的像素坐标中，格式为 xywh_pixels。
- snapshot JPEG 仍校验来源白名单、摘要、真实尺寸和框范围，DPAPI 加密保存。场景证据明确没有人员 person_id/track_id，不虚构人物编号。
- accepted=false 可能表示累计不足、重复、过期、缺图或没有新的通知，不能解释为“没有火焰”。

如队友通过现有 analyses 轮询返回 fire 结果，主项目也提供分发支持；对方现有 Schema 是否允许场景火焰结果仍需双方确认，不能假定已兼容。最小接入方式为上述独立 POST 入口，不需要先改人员交互结果的 Schema。

## 证据、时间与去重

第一次单源通知与双源升级分别只有一个事件 ID，保持同一 incident_id。通知保留硬件观测时间、视觉观测时间和图片自身时间。视觉先报警而硬件随后报警时，可复用该事件中已保存的视觉证据，图片时间不会改成新的硬件时间；它是该事件的历史视觉证据，不宣称与后来的硬件观测严格同帧。

持续视觉报警只更新信号，首次视觉证据和需要升级时才抓取图片，避免每帧重复落盘。群机器人仍使用持久化队列、签名、超时、重试和幂等键；Webhook 超时后的外部重发仍可能重复。

旧硬件事件在首次确认区域映射时可沿用原 incident_id。既有旧式硬件恢复逻辑在尚未映射时保持兼容。覆盖范围改变、一个摄像头覆盖多个局部区域、不同源事件早已分别生成后再改映射，不属于自动无损合并保证；应在现场联调前确定映射并核查旧未关闭事件。

## 实现与验证

- fire_fusion.py：持久化分源信号、覆盖区域匹配、双源升级与恢复。
- visual_fire.py：场景结果、真实时间域、连续不同帧和快照证据校验。
- safety_incidents.py：本地硬件分发，断线/旧观测不能解释为恢复。
- alert_evidence.py：无人物的场景证据保存。
- management_api.py：鉴权结果入口与轮询分发；已有人员/防护入口兼容。

68 项相关回归测试通过，其中新增火焰测试 14 项，覆盖两个报警顺序、重复源帧、低置信度、缺图、错误框、未知时间域、分区不合并、硬件 5 秒新鲜度、断线、双源恢复、旧事件映射、三群图文及相机越权；前端 JavaScript 语法检查通过。测试不使用真实串口，不向真实飞书群发送消息。

正式启用前仍需：确认视觉模型结果来源与接口、确认设备/摄像头物理覆盖区域、验证权威时间和真实快照索引、计划加载新版服务，并完成真实画面和明确标识的现场演练。当前运行服务未重启，不能据此声称现场双源报警已启用或真实模型准确率已验证。
