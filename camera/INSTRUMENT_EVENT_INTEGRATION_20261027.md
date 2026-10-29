# 仪器事件接入与 2026-10-27 恢复记录

已接入的链路：原始同帧输入 → 人员平面位置候选 → MiniCPM 原有判断 → 原有持久会话 → 本机通知管理后台与展示网页 → 飞书实验人员群和负责人群。管理员群仍使用原有紧急告警路由。

## 本次修改

- 新增 instrument_event_sync.py：从接收端的持久会话同步，按原 session_id 和 started/ended 稳定编号幂等入库、入投递队列。所有历史操作会话补入后台；不把旧操作批量当作新通知发送。
- 本机 frontend_server 从 Windows 凭据管理器读取已有接收 Token，连接 127.0.0.1:8765。Token 不进入配置、网页或日志。
- 新增 /api/instrument-events 和展示端仪器记录表；主页面提示区显示真实保存的开始/结束/时长。未结束操作不随页面刷新编造时长。
- 飞书现有实验人员群和负责人群增加 instrument.started / instrument.ended；已有持久队列保留重试和状态。
- 集群 SQLite 3.51.0 的 WAL 并发打开/关闭阻塞，使用 sqlite_connection_guard.py 同时保护 open / close；自动 GC 保持开启。对应集群 bootstrap.py 已部署，本地 scripts/cluster_joint_entrypoint.py 保存相同启动方式。并发压力测试 12 线程、3000 次 open/read/close 通过，耗时 0.299 秒。SQLite 官方修复说明：https://sqlite.org/releaselog/3_51_2.html 。
- 10月27日发现历史接口随输入增长超时：已有约 39480 条 handoff、244 条判断，网页最后同步停在北京时间 02:50。事件接口原先每次扫描全部历史帧 payload。现改为所有持久判断仍完整用于重建操作会话，仅“未操作人员”补充列表读取最近 200 帧。历史图像、判断、开始/结束记录均未删除。修复后事件查询样本约 0.094 秒。
- 修改仅在兼容副本、独立集群副本及队友现用应用的接入层；原部署包保留。Prompt、仪器标定、地图候选与模型开始/结束判断规则未修改。

## 已通过

- 仪器会话同步 4 项、既有通知 19 项、通知状态/新路由 2 项，共 25 项相关测试通过。通知路由测试已按用户要求更新：人员进入保持静默，仪器开始/结束通知两个现有群。
- 额外 member_integration 16 项通过。
- 真实持久会话在接收器、后台和展示页可见；2026-10-26 的真实开始/结束通知已返回 sent。
- 2026-10-27 09:36:02，cam_2 / person_13938 / instrument_003（超声波清洗机）产生新的模型判定开始；后续判断有效时间更新到 09:36:10，并同步后台。该开始事件向 experimenters、managers 各投递 1 次，均 sent、无错误；不是人工插入的测试事件。
- 管理后台数据库 quick_check 返回 ok。独立查询验收详情见 runtime/instrument-integration-check-20261027.json。
- 浏览器实际检查事件页已显示今天的新事件，并选择 person_13938 查看对应操作记录。

## 未通过或不能据此确认

- 尚未取得上述当前会话的有效结束判断；不补造结束或最终时长。
- 这次恢复不能证明夜间全程无中断、无漏记录。历史接口确实发生过同步中断；过程不能记为 24 小时稳定运行通过。
- 自动配对和 MiniCPM 的现场判断准确性仍需独立现场验收；链路投递成功不等于操作判断正确。
- 额外前端分离回归 42 项中 40 项通过，2 项现有配置断言失败：cam_2 当前使用 bytetrack_guarded_lab.yaml，测试仍期望 bytetrack_lab.yaml；现有地图重叠 reid 阈值与测试期望不同。本次未修改这两份识别/标定配置，也未为了通过测试回改它们。

## 入口与保存位置

- 事件页：http://127.0.0.1:8765/demo-control
- 展示页：http://127.0.0.1:5173/display
- 管理页：http://127.0.0.1:5173/management （继续使用已有维护认证）
- 后台会话表：camera/runtime/management.sqlite3 的 instrument_sessions；事件和投递表 events/outbox。
- 当前判断数据库在已迁移的独立集群副本 /cluster/dataX/xqyi/joint-migration-20261026/8765/runtime/handoffs.sqlite3。本机兼容副本 runtime 的旧数据库不能当作迁移后的最新数据库。
- Windows 三个常驻任务继续运行：Lab Interaction Cluster Bridge、Lab Joint Frame Recorder、Lab Safety Frontend Watcher。依赖机器、网络与当前用户登录。
- 接入前备份：camera/runtime/instrument-sync-backup-20261026-210316；本次查询优化前服务器备份位于兼容副本 runtime/server.before_event_query_20261027.py，集群有对应同名备份。

## 2026-10-27 09:41 完整连接复核

- 使用本机现有维护认证实际请求管理后台 /api/management/events 与 /api/management/status，均成功；没有绕过后台认证。
- 接收端、后台展示接口的 32 段会话编号集合相同；逐段比对人员键、摄像头、仪器、开始、结束、时长字段，全部一致。
- 完整真实会话 7ef25bccb442955ebb5b321e：cam_2 / person_13498 / 精密电子天平，2026-10-26 北京时间 21:13:57.775 至 21:14:02.573，时长 4.798 秒。管理后台的开始与结束事件字段与源会话一致；这两个事件向 experimenters、managers 的 4 次投递全部 sent，attempts=1，error=null。
- 当前累计：32 个开始事件、12 个结束事件已保存后台；新增通知累计 10 次开始投递、2 次结束投递均 sent。历史补录不会批量重发旧通知。
- 新增复核证据：runtime/instrument-connection-final-20261027.json。本次没有发送人工生成的仪器事件或额外群测试消息。
- 连接复核通过不覆盖此前凌晨中断，也不代替现场识别准确率或全天连续稳定性验收。
