# 人脸识别与跨摄跟踪检查（2026-10-15）

## 范围与证据

只读检查 15001 /api/state、/api/mtmc/events?limit=300，6次状态采样约10秒；未修改算法、配置或重启服务。原始检查证据：runtime/identity-audit-live-20261015.json。采样时三路均无有效人员轨迹；不能据此重现用户刚才的某次错配。

最近300条保留事件的源端时间为10月15日15:35:55.149至15:36:25.505，非此次采样新产生。291条associate中：目标不在入口门区219条、源不在出口门区38条、衣服明暗冲突21条、ReID低于中等9条、无候选3条；另1条同摄恢复成功。计数为候选关联事件，不是人数、独立失败次数或成功率。

## 已确认运行状态

- mtmc-2.1.5，OSNet-x1.0 MSMT17，512维；当前并非早期可疑ReID权重。
- cam_1、cam_2 face_recognition_enabled=false；仅cam_entrance=true。8名人员各60份ArcFace图库特征，无缺失ID。
- 三路tracking_diagnostics.tracker_config均为config/bytetrack_lab.yaml，未指向bytetrack_guarded_lab.yaml。前者无guarded_lost_low；按当前YoloPersonTracker构造逻辑，Lost低分恢复分支不启用。不能将隔离验证代码视作已部署功能。
- 人体检测阈值cam1=.2，cam2/入口=.1；推理步长3/1/3。普通tracker高分阈值.25、低分.1、buffer60。buffer不是可靠的60帧真实秒数保证。
- 入口累计尝试71346、检测到人脸45957、known结果223。known计数先于3票确认，不等于223人，不是人脸准确率。当前缺分阶段拒绝原因统计，无法归因到单一人脸门限。
- 本次状态人体inference_ms约7—26ms；空场不能推断有人或多人时的耗时，更不能代表ArcFace单独耗时或端到端延迟。三路历史重连数很高，但10秒采样增量均0；未证明当前持续断流。

## 失效链与代码依据

### 1 人体框不稳定引发局部ID重建

YoloPersonTracker.track：普通分支调用model.track，局部分配不使用本项目OSNet。YOLOv8n全身框遇遮挡变半身框、位移/重叠突变，几何关联可能断开。低于高分阈值的检测并不保证重新激活Lost。10月10日录像已有实证：15.806秒轨迹39进入Lost，16.003秒仍有约.23检测但未接回，见OCCLUSION_REPLAY_FIX_20261010.md。只是延长buffer不改变低分关联对象。

### 2 局部断轨会削弱人脸累计证据

app.py CameraWorker.process：缺失local超过track_hold_seconds后clear_identity、face_store.forget_track、coordinator.forget_local。当前每路hold为1.5秒。face_identity.py IdentityStore.predict按(camera_id,track_id)积累特征；app.py accept_face_result按local投3票。新ID不能无条件继承旧票，需全局恢复证明。

select_face_target每次只挑一人；仓库入口face_interval_seconds=.8，连续成功理想从首票到第三票需1.6秒，加首次调度最多约.8秒。实际运行状态未导出该间隔，因此该计算是本地配置/代码结论，不是在线识别延迟测量。质量失败、轮询多人、ID重建都会增加等待。

face_identity.py：尺寸/亮度/对比/清晰度质量门，ArcFace相似度.55、第一第二名差距.12、跨帧连续性.65；8人均有图库，不能继续归因为图库为空。质量及分数的真实拒绝分布仍缺证据。

### 3 图像门区硬约束在外观打分前拒绝

association.py AssociationEngine.evaluate：衣服明暗不一致先拒绝，然后成对路线出口/入口门区，再做ReID。近期cam1→cam2 secondary有160条入口门区拒绝，cam2→cam1 main有55条。即使OSNet像同一人，也无法绕过前面的硬约束。事件不足以证明这些候选全是同一人，需要对齐原图再评估门区和锚点，不能直接删除限制。

### 4 身体外观证据不足和模板保护

yolo_person_tracker.py associate_tracks_batch：人体重叠IoU≥.25冻结模板刷新；partial_body不更新全身模板，其他样本还需质量≥.45，刷新间隔.8秒。冻结可防止把两人混合裁剪写进图库，但新生遮挡框可能没有干净模板。OSNet为整身外观向量，不具备部件可见性匹配，半身与全身差异会降低分数。现有高相似度.75、中等.52、差距.08，多帧确认并非仅按距离合并。

### 5 身份恢复对跨摄链路依赖较强

室内不开人脸，入口身份一旦丢失且跨摄不接回，只会显示匿名person_N。单一注册身份锁会将竞争轨迹保持匿名；正确做法是找回对应全局轨迹，而非取消唯一身份约束。地图geometry_fusion_allowed=false，近期拒绝为图像门区；没有证据归因于地图米制距离标定。

### 6 历史断流可能放大重建

app.py capture_generation变化后reset_processing_state，清局部跟踪与人脸状态。累计重连高可作为需核查的历史风险，不能仅凭累计值证明本次断轨原因。最新帧模式的dropped_frames也包括主动跳过旧帧，不等于网络丢包。

## 修改优先级（本轮仅建议）

1. 固化线上实际模型/源码/配置摘要；隔离重放Lost低分修复，验证身份与拒绝样本后再部署，不能只切开关。
2. 把事件对应源图像门区、实际框/锚点、候选ID画出来，修正具体不覆盖路径或框底锚点不稳定，不放开全部摄像头随意合并。
3. 增加人脸阶段耗时和拒绝原因、框像素、前两名分差及轨迹存活统计；据此设计识别窗口内多帧采样/重试，不盲目放宽姓名阈值。
4. 局部遮挡关联使用干净模板、可见部件及运动证据；新局部待确认与身份释放解耦，证据不足保持未知。
5. 单人连续路线先验收局部/全局/姓名三层，再做双人交叉、同色衣服与遮挡多人。缺逐帧真值时不报告IDF1/HOTA或多人已解决。

## 算法参考

- ByteTrack官方实现：https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py
- OSNet官方模型资料：https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md

算法资料用于核对机制；本项目失败原因以运行事件和录像证据为准。
