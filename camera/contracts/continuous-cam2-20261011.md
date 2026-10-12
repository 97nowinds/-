# cam_2 高清连续发送

本阶段明确开启图片交接发送，不开启接收方自动分析，也不把未确认的同步、操作区或画幅验收标记改为已完成。

启动 `camera/scripts/start_hd_sender.ps1`，在本机隐藏令牌输入框填写接收地址和新令牌。
令牌仅传入发送进程环境，不写配置、命令行或日志。

发送进程为 `camera/continuous_handoff.py`，每 1/3 秒从现有图片服务读取当前 cam_2 冻结快照。
同一个源帧编号不重复发送；不读归档样例，不发送 cam_1。
提交前检查 JPEG SHA-256、实际尺寸、8 MiB 上限和图片有效期。
事件仍为 `lab-handoff-1.2`，人员框已经是高清图坐标，不再放大。
人员和地图观测仅来自这一份冻结快照；无人员时发送空 tracks，缺少地图观测时保留缺失。
这些情形不等于“没有交互”。

真实时间保持 source 域，sync_status=unavailable，conversion_applied=false；
hd_alignment_status=pending，zone_revision 仍按配置保留实际值。
只有对方实际返回成功状态和 receipt_id 才计为成功提交。
失败后取新的最新源帧，不建立重试积压队列，也不为了补帧循环发送旧图。

状态文件：`camera/runtime/continuous-cam2-status.json`。
记录实际发送频率、最近 90 张的源帧编号、人员编号、地图观测数量、图片大小、
本机取图耗时、请求 ACK 耗时及失败数。
`backlog=0` 表示本机无发送队列，不代表接收端模型队列无积压。
最近实际请求保存在 `camera/runtime/continuous-cam2-last-event.json`，不包含认证令牌。

JPEG 已实测约 1.01 MiB；3 帧/秒、60 秒保存约需 182 MiB。
因此当前图片缓存上限调整为 256 MiB，仍有界；单张取图上限保持 8 MiB。
人工归档样例的到期时间及字节不变。

关闭时按状态文件中的 pid 停止发送进程；仅停止发送器，不停止摄像头和图片服务。
