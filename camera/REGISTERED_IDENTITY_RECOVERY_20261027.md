# 注册身份重新绑定

2026-10-27，北京时间。已部署至 15001 主服务。

## 目标与规则

允许 local track_id、global person_x 改变。新轨迹通过人脸确认后绑定同一注册 person_id，
不要求合并前后全局轨迹，不通过脚点接近分配姓名。

1. 保留三次人脸匹配投票，原有相似度/差距/人脸质量门槛不变。
2. 确认结果保存内部证据：确认的单调时钟、票数、来源 camera_id 和 local track_id。
3. 对另一个轨迹占用的注册身份，必须确认新轨迹有本相机新鲜人体观测，且人脸证据不超过 2 秒。
4. 原归属任一相机在最近 2 秒仍有真实人体观测，则拒绝转移，保持未确认。
5. 原归属已无新鲜人体观测，允许接回注册身份，即使旧 global track 仍留在缓存。
6. 不合并两条轨迹，不新建 redirects；已转移的旧姓名缓存不能再显示或抢回姓名。
7. 暂时身份锁冲突会清除可发布姓名，但保留该局部轨迹的人脸投票；下一次真实人脸匹配可重新申请。
   真正轨迹丢失时仍清除其投票，新局部轨迹不会继承旧票。
8. 未获全局身份锁之前，不将三票候选提前分享给 handoff。获准后沿用原有 refresh_handoff 发布。

新鲜度依据是 coordinator 的人体观测 last_seen，不能用持续刷新的 identity.updated_at 代替。
绑定调用在本次人脸推理之后读取 time.monotonic()，避免使用帧处理开始时间判断新证据为“来自未来”。
内部人脸证据不是画面采集时间，不进入交接 Schema 的 observed_at 字段。

## 修改

- face_recovery.py：独立、可测试的证据和归属检查。
- yolo_person_tracker.py：set_identity 允许可靠人脸接回无新鲜观测的旧归属。
- app.py：记录确认来源；绘制之前清除失效姓名缓存；冲突时保留投票，先获锁再分享身份。
- face_attempt_audit.py：增加 completed_at 和匹配通过时的注册 person_id，便于区分候选和身份归属。
  completed_at 为处理完成时刻，不是采集时间；不保存图像或特征向量。

本轮没有部署先前失败的 tracklet recovery 候选，也未替换线上 ReID 关联实现。
三路高清人脸输入、三路识别和 2K 录像配置继续生效。15002 副本未改变。

## 验证

集群隔离执行 86 项单元/回归测试全部通过，其中 worker 方法从候选 app.py 的 AST 提取，
防止导入主应用触发服务或模型初始化。包括：

- 残留旧轨迹接回姓名，两个全局轨迹仍分别存在；
- 任一路旧人体观测新鲜时拒绝抢占；
- 错相机、错局部编号、少于三票、过期/未来人脸证据均拒绝；
- 新人体观测过期或缺失时拒绝；
- 旧缓存不得夺回已转移姓名；
- 锁冲突保留票但清除姓名，真实轨迹丢失清除票；
- 三票候选不提前传播给 handoff。

对刚录制的三路 2K 素材，复用集群 GPU 回放得到的同帧人脸结果，
重放基线和候选的实际身份绑定方法。局部轨迹各分配独立全局编号，不模拟 ReID/地图/handoff。
两版均在 cam_2、录像约 65.927 秒将局部 13 绑定至注册 person001，均为一段成功绑定，
没有新的恢复事件，也没有新增拒绝。

这个素材中没有第二段满足三次确认的新轨迹，所以不能宣称“实录恢复次数已提升”。
该回放验证已有确认未退化；新增残留归属释放分支由单元与集成场景验证，仍需现场断轨后再次露出可用人脸验收。
注册身份未知或证据不足时继续保持未知。

## 上线与回退

只对实际运行文件做定点修改，部署前核对与备份一致，避免覆盖其他维护。
主服务已重启并验证 cam_1、cam_2、entrance 均 running、face_recognition_enabled=true、ready=true。

集群路径：
`/cluster/home/xqyi/lab-safety-host/camera/runtime/registered-recovery-20261027/`

- baseline/：部署前 app.py、yolo_person_tracker.py、face_attempt_audit.py；
- candidate/：已测试候选和测试文件；
- unit-tests.json、recorded-binding-replay.json、deployment.json：验证与文件哈希。

回退前应比较 deployment.json，确认没有后续维护改动，再恢复 baseline 三文件、移除本轮新增 face_recovery.py，
重启 lab-safety-api.service。不得覆盖此后无关修改。
