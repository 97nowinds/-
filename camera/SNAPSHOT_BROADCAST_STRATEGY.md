# 实验室分钟级截图与人物播报接入策略

状态：方案已保存，尚未实施
计划使用时间：2026-09-14 下午
默认时区：Asia/Shanghai（UTC+08:00）

## 1. 目标

每个自然分钟保存一次能够回溯现场状态的证据包，内容包括：

1. 实验室实时位置地图；
2. 当前三路监控画面；
3. 环境与火情状态；
4. 后续行为识别接口提供的“谁、在哪里、正在做什么”；
5. 与上述状态对应的播报文字。

截图必须与结构化数据使用同一个 `snapshot_id` 和时间戳。不能只保存网页图片，否则后续无法可靠检索某个人当时的位置、动作、置信度和来源摄像头。

## 2. 推荐输出

每分钟产生一个 PNG 和一个同名 JSON：

```text
data/snapshots/2026-09-14/15/
  20260914T150300+0800.png
  20260914T150300+0800.json
```

PNG 使用固定版式：

```text
┌──────────────────────────────┬──────────────────────────────┐
│ 实验室实时位置地图           │ 人物位置/行为与播报           │
│ 人员点位、轨迹、区域名称     │ 2号王某：主通道，正在行走     │
│                              │ 5号李某：实验台旁，动作未知   │
├──────────────┬───────────────┼──────────────────────────────┤
│ Camera 1     │ Camera 2      │ Entrance Camera              │
│ 时间戳       │ 时间戳        │ 时间戳                       │
├──────────────┴───────────────┴──────────────────────────────┤
│ COM3 在线 · 26.0℃ · 43.4%RH · 火情安全 · 数据质量提示       │
└─────────────────────────────────────────────────────────────┘
```

建议输出分辨率为 1920×1080；保留原始画面比例，不拉伸。人物框、地图点位和播报条目使用一致的身份颜色。

## 3. 时间与一致性规则

- 调度锚定自然分钟的 `:00`，不是“任务完成后再等 60 秒”。
- `snapshot_id` 格式为 `YYYYMMDDTHHmmss+0800`，同时作为幂等键；同一分钟重复触发不得生成多份正式记录。
- 先冻结一次 `/api/state`，再获取各摄像头最新帧和行为结果。
- 每路画面必须记录自身的 `frame_captured_at`，不能把网页截图时间冒充摄像头采集时间。
- 默认允许画面与地图状态相差不超过 2 秒；超过时仍可保存，但 JSON 标记 `synchronized: false`，PNG 显示“时间未对齐”。
- 行为结果设置有效期，默认 5 秒。过期行为显示“动作未知”，不能沿用上一分钟的动作。
- 单路摄像头失败不阻断整包保存：对应区域显示“画面不可用”，JSON 记录失败原因。

## 4. 结构化数据契约

建议的分钟记录：

```json
{
  "schema_version": "snapshot-1.0",
  "snapshot_id": "20260914T150300+0800",
  "captured_at": "2026-09-14T15:03:00.217+08:00",
  "timezone": "Asia/Shanghai",
  "synchronized": true,
  "max_skew_seconds": 0.8,
  "environment": {
    "temperature": 26.0,
    "humidity": 43.4,
    "flame_alarm": false,
    "status": "online"
  },
  "people": [
    {
      "person_id": "person_002",
      "display_name": "王某",
      "track_id": "global-17",
      "location": {
        "zone_id": "main_aisle",
        "zone_name": "主通道",
        "x_m": 4.25,
        "y_m": 2.10,
        "confidence": 0.92
      },
      "activity": {
        "code": "walking",
        "label": "行走",
        "confidence": 0.84,
        "observed_at": "2026-09-14T15:02:59.810+08:00",
        "source_camera_ids": ["cam_1"]
      },
      "broadcast_text": "2号王某位于主通道，正在行走。"
    }
  ],
  "cameras": [
    {
      "camera_id": "cam_1",
      "frame_captured_at": "2026-09-14T15:03:00.020+08:00",
      "frame_age_seconds": 0.2,
      "status": "ok"
    }
  ],
  "artifacts": {
    "image": "20260914T150300+0800.png",
    "metadata": "20260914T150300+0800.json"
  }
}
```

## 5. 后续行为接口约定

行为服务只负责提供观察结果，不直接拼接页面。建议统一成：

```http
GET /api/activity/state?at=2026-09-14T15:03:00.217%2B08:00
```

最少返回：

- `person_id` 或 `track_id`；
- `activity.code`、`activity.label`、`activity.confidence`；
- `observed_at`；
- `source_camera_ids`；
- 可选的检测框和证据帧编号。

对齐策略：优先用全局 `person_id`；未确认身份时使用 MTMC 全局 `track_id`。位置以现有 `/api/state` 的 `floor_map.people` 为准，行为以时间上最接近且未过期的行为观察为准。

播报句子由本系统根据结构化字段生成，避免第三方接口随意返回不可控文案：

```text
{人员名称}位于{区域名称}，{行为描述}。
```

低置信度必须明确表达不确定性：

- `confidence >= 0.75`：正在操作设备；
- `0.50 <= confidence < 0.75`：疑似正在操作设备；
- `confidence < 0.50` 或结果过期：动作未知。

位置和行为不得混成一个模型结论。例如地图确认“人在主通道”，行为模型只确认“弯腰”时，应播报“位于主通道，检测到弯腰动作”，不能推断为“正在捡东西”。

## 6. 建议新增的本地接口

```http
POST /api/snapshots/capture
GET  /api/snapshots?date=2026-09-14
GET  /api/snapshots/{snapshot_id}
GET  /api/snapshots/{snapshot_id}/image
GET  /api/snapshots/latest
```

`POST /api/snapshots/capture` 同时供定时器和人工测试使用，返回 `snapshot_id`、PNG/JSON 路径、同步质量和缺失来源。接口内部使用锁和分钟幂等键，避免定时任务与人工点击并发写入。

## 7. Hooker 成员交接服务

后续通过 Hooker 将截图、人物位置、行为播报和异常状态交接给其他成员。Hooker 只承担“投递、认领、确认和审计”，不参与摄像头取帧、人员识别或截图生成，因此 Hooker 暂时不可用时不能影响本地每分钟证据包的保存。

推荐链路：

```text
每分钟调度
  → 本地生成 PNG + JSON
  → 写入待交接队列（outbox）
  → HookerAdapter 投递 snapshot.ready
  → 成员/团队认领
  → Hooker 返回 receipt_id 和确认状态
  → 本地记录交接审计
```

建议的 Hooker 事件信封：

```json
{
  "schema_version": "hooker-handoff-1.0",
  "event_id": "snapshot.ready:20260914T150300+0800",
  "event_type": "snapshot.ready",
  "occurred_at": "2026-09-14T15:03:01.102+08:00",
  "producer": "lab-camera-host",
  "handoff": {
    "target_team": "待配置",
    "assignee": null,
    "priority": "normal",
    "summary": "2号王某位于主通道，正在行走。",
    "requires_ack": true
  },
  "snapshot": {
    "snapshot_id": "20260914T150300+0800",
    "image_url": "待配置的受控访问地址",
    "metadata_url": "待配置的受控访问地址",
    "synchronized": true,
    "people_count": 1,
    "alarm": false
  }
}
```

交接规则：

- 正常分钟记录发送 `snapshot.ready`，默认普通优先级；
- 火情、设备离线或关键接口连续失败发送独立的 `snapshot.alert`，不能依赖下一次分钟记录；
- 行为接口晚到并修正结果时发送 `snapshot.activity_updated`，引用原 `snapshot_id`，不覆盖原始审计记录；
- `event_id` 必须幂等，网络重试不能让其他成员收到重复任务；
- Hooker 成功接收后保存 `receipt_id`、接收时间、目标团队和最终认领成员；
- 状态统一为 `pending`、`delivered`、`acknowledged`、`failed`；只有收到 Hooker 确认才算完成交接；
- 投递失败采用指数退避，并保留本地 outbox；不能因为进程重启丢失未交接事件；
- PNG 文件较大时只发送受控下载地址和摘要，不直接把 Base64 图片塞入事件；
- 接口密钥只从环境变量或系统凭据读取，不能写入代码、JSON 证据包或 Git。

建议预留本地接口：

```http
GET  /api/handoffs?status=pending
POST /api/handoffs/{event_id}/retry
POST /api/hooker/callback
GET  /api/handoffs/{event_id}
```

下午正式对接前必须向其他成员确认以下 Hooker 契约：

1. “Hooker”的准确产品/项目名称及接口文档；
2. 事件接收 URL 和回调 URL；
3. OAuth、API Key、HMAC 或其他认证方式；
4. 必填字段、事件名称、附件大小和下载方式；
5. 目标成员/团队标识格式；
6. 成功响应、确认回执及重试语义；
7. 测试环境与正式环境地址；
8. 人脸、姓名、位置等敏感数据允许传递的范围。

在上述契约确认前只实现 `HookerAdapter` 接口和本地假实现，不向未知地址发送任何真实实验室数据。

## 8. 实施架构

推荐新增 `SnapshotCoordinator`，运行在能够同时访问以下资源的本机服务中：

- 远端视觉后端：当前 `http://127.0.0.1:15000/api/state`；
- 三路监控帧：现有 `/video/<camera_id>`，实施时最好新增单帧 JPEG 接口 `/api/cameras/<camera_id>/snapshot`；
- 本机环境硬件：当前 `http://127.0.0.1:5173/api/environment`；
- 未来行为接口：`/api/activity/state`。

合成工作不依赖浏览器窗口是否打开。前端页面可增加“最近截图”“立即截图”和“截图健康状态”，但定时保存由后台服务完成。浏览器全页截图只作为视觉回归测试，不作为正式证据生成方式。

组件职责：

1. `SnapshotScheduler`：自然分钟调度、补偿、停机恢复；
2. `SnapshotCoordinator`：冻结状态、拉取帧、时间对齐；
3. `ActivityAdapter`：屏蔽后续行为服务的具体协议；
4. `SnapshotRenderer`：绘制地图、监控格、播报和环境状态；
5. `SnapshotStore`：原子写入 PNG/JSON、索引和保留策略。
6. `HookerAdapter`：事件映射、签名、投递、确认回调和失败重试。
7. `HandoffOutbox`：持久保存尚未被 Hooker 确认的交接事件。

保存时先写入同目录临时文件，PNG 和 JSON 都完成后再原子改名，防止断电留下看似完整的半包数据。

## 9. 保留、隐私与安全

- 默认建议保留 30 天，具体期限在实施前确认；自动清理必须单独配置，默认不开启删除。
- 人脸原图和身份名称属于敏感信息；截图目录不得放入 Git，也不应直接暴露为无需认证的公网静态目录。
- 在 `.gitignore` 中加入 `camera/data/snapshots/`。
- 火情状态属于辅助监测信息，截图和播报不能替代现场蜂鸣器、LED、急停或认证安全联锁。
- 每次生成记录审计字段：程序版本、接口版本、失败来源、最大时间偏差。
- Hooker 交接日志记录事件和回执，但不得记录访问令牌、签名密钥或完整认证头。

## 10. 下午实施顺序

1. 确认截图目录、保留天数、输出分辨率和是否显示真实姓名；确认 Hooker 的准确接口契约；
2. 给远端后端增加单帧 JPEG 接口，并携带真实帧时间戳；
3. 实现 JSON 证据包与原子存储；
4. 实现地图和三路画面的 1920×1080 合成；
5. 增加每分钟调度、幂等和失败降级；
6. 接入临时 `activity=unknown` 适配器，先跑通完整链路；
7. 对接真实行为接口并生成播报；
8. 增加列表、最新截图和人工触发接口；
9. 先用本地假 Hooker 验证 outbox、幂等、确认和重试，再切换测试环境；
10. 连续运行至少 10 分钟，核对应有 10 个唯一时间槽、20 个文件，且无重复和半包；
11. 模拟一台摄像头断线、行为接口超时、COM3 断开、Hooker 超时和火情告警，检查截图降级、播报措辞与交接重试。

## 11. 完成判定

- 连续 60 分钟应得到 60 个唯一证据包；
- 每包 PNG 与 JSON 的 `snapshot_id` 一致；
- 人物点位、摄像头帧、行为观察最大时间偏差可计算；
- 人物身份不确定、动作低置信度、数据过期时不会做确定性播报；
- 任一单独数据源失败不会导致整个分钟记录丢失；
- 页面、接口和磁盘索引能够定位到同一张截图和同一条播报记录。
- Hooker 重复投递不会创建重复交接，服务恢复后能够补发 outbox 中的记录；
- 每个要求交接的事件都能追溯到 Hooker 回执、认领成员或明确的失败状态。
