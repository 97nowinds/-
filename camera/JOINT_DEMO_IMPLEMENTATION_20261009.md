# 最小联合展示：发送端实施与交接

本轮范围为 `joint-demo-minimal-r1`，取代旧长方案的全量实施要求。
本仓库是主项目/发送端；192.168.31.189 上的 MiniCPM、高清缓冲、区域候选和接收 Schema 不在本仓库，不能以本地测试代替对方实现或现场验收。

## 已实现

- 检测进程在成功的实际检测帧上冻结 JPEG、xywh 框、锁定身份、同轨迹的当次定位观测、地图版本和原始时间。跳过缓存框、held 框和检测失败；冻结默认关闭，通过 `LAB_HANDOFF_FREEZE_ENABLED=1` 开启取证。
- JPEG 使用处理后的完整画幅，没有预览叠加；检测框与 JPEG 同尺寸，因此发送时无需再次缩放。`source_dimensions` 保留原始解码尺寸。当前处理 resize 是完整画幅等比例缩小，未引入 letterbox；摄像头两条流是否同画幅仍需实测。
- `GET /api/handoff/frozen/{camera_id}`：5175 从检测后端获取冻结证据，保留在本机不可变缓存中。这是发送端内部的证据读取，不是新增接收端业务接口。
- `GET /api/handoff/snapshots/{snapshot_id}.jpg`：5175 的后台认证 JPEG 路由。首次放入缓存时固定 SHA256 与有效期，保留 60 秒；重读不续期。容量满时拒绝新证据，不提前淘汰承诺有效的图片；过期 ID 不复用。
- `POST /api/handoff/time-sync`：5175 提供独立后台认证探测。请求严格只接受 `probe_id` 和带时区的 `client_sent_at`，响应原样回显，加 `authority_clock`、`authority_received_at`、`authority_sent_at`。客户端记录第四时间戳，独立估算偏移和往返不确定度。检测源已有周期同步客户端，接收端必须另行同步自己的高清进程。
- 发送器支持 cam_1/cam_2 独立线程，复用现有 POST /api/handoffs；形成 1.2 事件前，验证冻结同步映射、确认尺寸/对齐版本/区域版本、实际 JPEG 路由和哈希，以及接收端提供的 Draft 2020-12 Schema。缺任何条件不发送。
- 重试保留同一个 event_id、snapshot_id、完整事件与 JPEG。结果索引在首次 POST 前建立，以处理结果先于 ACK。图片有效期、同步有效期或推理开始帧龄到期即停止旧事件重试。
- 新版查询显式透传 `contract=lab-interaction-results-1.2`，要求响应回显 `result_contract`，并校验 `analyses[].status` 与布尔/null 关系。无参数请求保留旧分支。
- 新版身份只从已发送的同相机/快照/轨迹索引恢复。匿名轨迹保持匿名；未匹配结果不进入当前正状态。
- 有效结果按首次发布后最多 5 秒和本机单调时钟过期。重复查询不能延长；迟到的旧正结果不能覆盖已显示的更新结果。新版 analyses 记录写入本机 `camera/runtime/joint-demo-analyses.jsonl`，过期后的历史判断仍保留，当前投影返回 expired/null。
- 页面通过 `LAB_MEMBER_RESULT_CONTRACT=lab-interaction-results-1.2` 选择新版，显示匿名轨迹、证据时间与 pending/unavailable/error/stale/expired，空列表不显示成模型判否。

## 共同参数表

统一文件：`camera/config/joint_demo.json`。检测源与主项目必须部署同一份确认配置。

| 参数 | 当前值 | 状态 |
|---|---:|---|
| JPEG 保留 | 60 秒 | 满足至少 30 秒约定 |
| 位置/框最大时间差 | 1 秒 | 接收侧现状，未现场验收 |
| 高清配对最大时间差 | 0.5 秒 | 接收侧现状，需对方执行 |
| 推理开始最大帧龄 | 2 秒 | 接收侧现状；发送端也停止超龄重试 |
| 业务结果 TTL | 5 秒 | 首次发布起，轮询不续期 |
| 同步频率、有效期、最大不确定度、跳变阈值 | null | 等待双方确认；同步客户端不产生有效映射 |
| 结果发布最大年龄 | null | 等待实际对齐测试确认；不猜值 |
| 逐相机尺寸/对齐记录 | 空对象 | 必须填写实测 source_dimensions、jpeg_width/height、hd_width/height、alignment_revision 和 transform=full_frame_resize |
| zone_revision | null | 接收端确认区域文件的 SHA256，不能用地图版本替代 |
| receiver_schema_file | null | 接收端最终 1.2 Schema 的本地绝对路径 |
| alignment_verified / automatic_enabled | false | 对齐验收前保持关闭 |

时间转换只执行一次，`drift_ppm=0`，本轮不实现复杂漂移模型。短期映射有效期及不确定度必须按实测确认。业务到达延迟没有参与同步计算。

## 身份与仪器边界

第一轮 `track_id == global_id`，local_track_id 单独保留。检测进程冻结的是同相机投影观测，非之后独立 positions 查询中的平滑/保持点；其实际时间与地图版本留在证据里，不从最新地图点反向猜身份。

| 相机 | 接收端仪器 |
|---|---|
| cam_1 / lab_camera_view_2 | instrument_006 高速离心机，Tag 21 |
| cam_2 / lab_camera_view_1 | instrument_001 精密电子天平，Tag 41；instrument_003 超声波清洗机，Tag 31 |
| cam_entrance | 不发仪器自动分析 |

不添加磁力搅拌器，不生成未经确认的操作多边形；多人/多仪器歧义由接收端报告 unavailable，不能靠数组顺序解决。

## 启动与认证

5175 的两个读取权限分别使用 `LAB_HANDOFF_IMAGE_READ_TOKEN` 和 `LAB_HANDOFF_SYNC_READ_TOKEN`。也可通过 `--credentials-file` 读取仅在后台使用的 JSON：键为 `image_read_token`、`sync_read_token`。凭据不写入 URL、业务 JSON、浏览器或日志。当前本机凭据文件位于 `camera/runtime/handoff-credentials.json`（Git 已忽略 runtime）。通过私下后台配置把必要权限交给检测源和接收端，不通过聊天或浏览器传值。

`scripts/start_handoff.ps1` 启动 5175，默认后端 http://127.0.0.1:15001，可传 -Python 和 -BackendUrl。`GET /health` 回显 `service_revision=joint-demo-minimal-r1`，便于检查是否实际重载。

检测源需要后台注入两项读取凭据、共同参数文件和 `LAB_HANDOFF_FREEZE_ENABLED=1`，随后重载检测服务。发送前端需要 image 读取凭据、既有接收端业务 Token，以及明确的 `LAB_MEMBER_LIVE_PUBLISH_ENABLED=1`；配置中 automatic_enabled 仍需完成验收后才能打开。新分支发送前依赖 requirements.txt 中的 jsonschema。

## 仍需队友确认的差异

1. 新最小方案采用响应 `result_contract` 与 `analyses[].status`；旧 Word 是 `result_contract_version` 与 `analysis_status`。本轮按新最小方案，不静默把旧响应改造成新版。
2. 请提供最终严格 1.2 Schema、结果 JSON，以及共同参数表中的待确认值；本地发送适配器必须通过这份 Schema，不能用自己的宽松 Schema冒充已兼容。
3. 证据中原始 event formation 时间在检测端冻结，JPEG 的公开可取时间在 5175 首次缓存时记录，可能稍晚。若接收 Schema 仍沿用旧 Word 的 image_available_at 不得晚于 occurred_at 约束，需要统一这两个阶段的定义；本轮不伪造源时间来满足它。
4. 接收端需完成高清自身时间转换、已确认区域文件版本、三个不同历史帧约两秒窗口、人员不串人验证和新版响应。主项目不能代替它完成这些工作。

未经两路单人移动与两人不串人验收，不开启 automatic_enabled。现场三台仪器测试仍须确认模型原始输出与双方页面一致。

## 验证与限制

单元/HTTP 测试覆盖不可变性、60 秒保留、容量拒绝、过期 URL 不复用、读取权限隔离、四时间戳估算、一次转换、匿名身份、尺寸不匹配拒绝、重复轨迹拒绝、历史和业务 TTL 分离、迟到结果不反向覆盖。相关 Python 编译与页面 JS 语法检查通过。

工作区已有地图配置将 simultaneous_reid_threshold 从 0.95 改为 0.75，现有 test_approximate_map_enables_observed_entrance_overlaps 仍要求 0.95，因此该原有测试失败。本轮不修改该配置，也不以放宽阈值解决验收问题。

本轮不声称检测源、接收端已部署新版，也不声称 JPEG 的真实人员内容已完成远端拉取与高清对应验收。

## 本机实际运行核对

2026-10-09：5175 已重载为 joint-demo-minimal-r1（进程 34728），实测 health=200、认证 time-sync=200、错误读取权限=403、不存在的 JPEG=404。冻结证据读取为 503，真实 JPEG 尚未验证。

5174 旧进程 2964 的发送器仍启用，检查时为 idle_no_tracks，累计已发送 1667 帧。尝试重载 5174 并关闭旧发送器、同时更新 5175 的最终小改动时，被自动审批以 blocked by policy 拒绝，命令未执行。不能把磁盘默认关闭当作该旧进程已经关闭。

需要在本机停止旧 5174 进程并使用新版启动脚本重载；-DisableLivePublisher 现在不依赖 -ConfigureMemberIntegration。新脚本可通过 -PythonPath 指定 Python。重载前端时如需保留接收端查询，应继续通过原有后台方式注入 LAB_MEMBER_SERVICE_TOKEN，不从浏览器读取。5175 的最终源码也需再次重载；此前已验证的新路由仍在运行。

本轮最终相关检查为 76 项，75 项通过、上述既有地图阈值测试 1 项失败；Python 编译、JS 语法检查通过。单独新增的 11 项联合对接测试全部通过，含 ACK 丢失重试和旧响应不能冒充 1.2。
