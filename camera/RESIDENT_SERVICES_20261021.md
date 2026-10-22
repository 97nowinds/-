# 长期值守恢复记录（2026-10-21）

## 已核实的断开原因

- Windows 旧看守仅检查前端端口，未看守 SSH 隧道和摄像头推流。故前端仍能打开，但集群接口及视频源已断开。
- 集群 API 仍在运行，三路摄像头持续重连；转发服务没有源流，返回找不到视频流。未发现证据表明此次是模型崩溃或电脑关机。
- 旧前端重启没有保留管理通知启用参数，管理接口曾返回 503。
- 新配置窗口首次验证失败另有原因：新增 `-rw_timeout` 参数不受当前 FFmpeg 支持，尚未连接摄像头就退出。已移除，并真实验证三路画面；不能将这次错误归因于用户密码。

## 当前机制

Windows 任务 `Lab Safety Frontend Watcher` 运行 `scripts/resident_services.py`，管理 SSH API/视频隧道、三路视频推送、5173 前端及通知、5175 地图交接服务。

- 子进程退出后约 5 秒再启动；检查周期及网络连接会影响实际恢复时间。
- 推流 45 秒没有帧数进展时重启；API 连续三次访问失败时重建自有隧道；网页服务进行 HTTP 健康检查。
- Windows Job Object 在值守主进程异常退出时清理其子进程，防止遗留进程占用端口或重复推流。
- 计划任务没有执行时长限制，登录后自动启动，且每分钟重复尝试启动；服务已运行时忽略重复启动。最初仅使用“失败后重启”的配置在强制终止测试中未生效，已补上重复触发兜底。未承诺登录前启动。
- 接通电源时抑制自动系统休眠，允许熄屏；未改变电池模式的系统休眠策略。显式休眠、关机、断电和网络中断仍会影响服务。
- 密码在本机用当前用户 DPAPI 加密；有效配置为 `runtime/resident-camera-sources.dpapi`，待验证输入另存加密文件，后者不会用于正式推流。日志不记录密码或 RTSP URL。
- 视频传输使用 FFmpeg stream copy，不在本地增加深度学习推理。

集群 `systemd --user` 服务为 `lab-safety-api.service`、`lab-safety-relay.service`、`lab-safety-health.service`。已确认均为 active，用户 linger 已开启；退出 SSH 不会结束它们。HTTP 连续故障触发 API 重启，单纯缺视频不会误触发 API 重启。

## 访问与排查

- 监控：[打开监控](http://127.0.0.1:5173/?api=http%3A%2F%2F127.0.0.1%3A15001&remote=1)
- 通知后台：[打开后台](http://127.0.0.1:5173/management)
- 交接地图：[地图接口](http://127.0.0.1:5175/api/handoff/map)（继续遵守原接口鉴权）
- 状态：`runtime/resident-status.json`；轮转日志：`runtime/resident-services.log`。
- 读取状态时必须同时检查 `updated_at`：超过 20 秒未更新应视为失联，不能信任文件里遗留的 `healthy=true`。
- 状态中的 `external_listener` 表示端口由非本值守子进程占用，不能据此声称该服务已纳入看守；不会根据端口扫描盲目杀死其他进程。
- `healthy` 要求三路画面新鲜且自有组件都存活。它只表示采集和服务健康，不能证明识别准确、跨摄身份连续或告警检测已接入。

## 验证

三路配置均通过真实画面探测并保存，三路后端状态恢复 running。已主动结束自有 SSH 隧道、前端和入口推流，均自动生成新进程并恢复三路实时采集；结果记录于 `runtime/resident-recovery-test.json`。

通知管理接口带原本管理员凭据返回 200，未鉴权返回 401。此过程未发送模拟的真实违规或火情消息，检测端尚未接入的限制仍保留。

补上重复触发后，第二次强制结束值守主进程的实测通过：旧子进程全部退出，新值守自动启动，约 50 秒后三路恢复健康。结果另存 `runtime/resident-owner-recovery-test.json`。恢复时间会随网络和摄像头重连而变化。

## 停止与回退

执行 `Stop-ScheduledTask -TaskName 'Lab Safety Frontend Watcher'` 停止 Windows 值守及其自有子进程；`Disable-ScheduledTask` 可取消后续自动启动。恢复使用 `Enable-ScheduledTask` 后 `Start-ScheduledTask`。不删除加密配置。

集群可分别 `systemctl --user stop lab-safety-health lab-safety-api lab-safety-relay`；取消自动启动使用对应 `disable`。恢复后检查摄像头流与 HTTP 状态。不要在新值守仍运行时叠加旧启动脚本。
