# 人员定位与仪器交互交接核对（2026-10-08）

## 范围与证据

检查了 `member_integration.py`、`frontend_server.py` 的 `_tracks/_publish_once`、`map_handoff.py`、`static/instrument_location.js` 和结果显示代码；为确认来源，只读检查 `app.py` 的采集/resize/JPEG/状态导出、`yolo_person_tracker.py` 的框导出、`floor_map.py` 的坐标和时间导出，以及 FFmpeg 推流脚本。未修改人物检测、跟踪、身份识别、地图定位或仪器区域算法。

接收端正式 JSON Schema 未出现在当前主项目中；已有代码标注 `lab-handoff-1.1`，接收端 `/openapi.json` 返回 404。不能声称已验证接收端所有字段约束或 additionalProperties 行为。本目录的 Schema 文件是补丁建议，不是接收端正式 Schema。三台仪器区域按用户现场确认接受，本次未重新标注。

当前已核验地图接口返回米制坐标、12×7 米、`map_id=lab-floor`；审查时布局版本为 `d9ea86e4bd80d086`。版本可能因布局更新变化，发送时不得硬编码。

## 1. 字段现状

| 字段 | 当前生成情况 | 实际缺口 |
|---|---|---|
| snapshot_id | 有，发送端构造并用于 image_url | ID 是发送实例标识，不是原始相机帧序号 |
| captured_at | 有，从单独读取的 camera.capture_received_at 获取，缺失时使用发送端 now | 不是随后从 MJPEG 读到的 JPEG 的已验证采集时刻 |
| occurred_at | 有，事件构造器直接等于 captured_at | 不代表回执或发送完成时刻 |
| camera_id | 有 | 必须与高清输入来自同一 camera_id |
| frame_width/frame_height | 有，解析发送 JPEG 的真实尺寸 | 与人物框坐标尺寸当前可能不一致 |
| tracks[].track_id/global_id | 有，旧逻辑两者都填 global ID | local_track_id 是真正本地 ByteTrack ID；不能把三种编号视作同义 |
| person_id/display_name | 字段有；原 live 逻辑恒为 null | 本次补齐：只从同相机 identities 中按 local_id、global_track_id、locked 状态唯一匹配；未知保留 null |
| bbox/bbox_format | 有，xywh_pixels | bbox 未按 960→640 的 JPEG 缩放同步换算 |
| track.observed_at | 有，但直接填 event 的 captured_at | 不是框实际检测帧的已确认时间；缓存框也可能被标成新时刻 |
| location.x_m/y_m/quality/observed_at | 有条件提供 | 定位缺失或时间不匹配时省略；之前位置时间取 person.observed_at，本次改为 position_observed_at 优先 |

字段存在不证明这些字段属于同一个实际采集帧。

## 2. 地图人员和交接轨迹对应关系

当前 `/api/handoff/map` 与 `/positions` 返回的 `people[].track_id` 是地图分组键；对于已识别人，它可能是注册身份分组键，不必等于一个相机的 local_id。跨摄关联应使用 `global_track_id` 或 `global_track_ids`。

`camera_tracks` 中：`camera_id` 为摄像头，`local_id` 为相机本地轨迹，`track_id` 是后端当前 global ID。旧 snapshot.frame 中 `track_id=global_id`，`local_track_id=local_id`。这是实际旧契约，不应按数组位置或显示姓名连接。`count_duplicate_of` 只用于人数去重，不是确认身份，不可拿它把另一个人的位置套到本轨迹。

当前地图到事件的位置连接：用事件 global_id 匹配人员 global aliases，取时间容差内的样本。未找到则省略 location。此次增加冲突/held/predicted位置过滤；同一 alias 对应多个地图人员行时不分配位置。全局关联本身是否正确仍依赖原系统，不能由导出层证明。

双方约定新语义后，建议：`track_id=str(local_id)`（仅在相机范围内有效）、`global_id` 继续使用跨摄 global ID、`person_id` 为已确认的注册身份。实时索引 `(camera_id,track_id)`；具体事件索引 `(camera_id,snapshot_id,track_id)`。ID 语义变更未启用，需双方共同切换。local ID 可在重新启动/跟踪重置后复用，历史结果必须含 snapshot_id。

位置坐标经过源码核对：floor_map 输出 x/y 直接以 width_m/height_m 表示米；页面按 x/width_m 和 y/height_m 映射 SVG，无 Y 轴翻转；map_handoff 只把 x/y 改名为 x_m/y_m；location 从同一 floor_map 坐标取值，无像素换算。原点左上、X 向右、Y 向下。地图为 approximate 标定，不是精确仪器测量坐标。

地图接口读取原始后端 floor_map。浏览器 `InstrumentLocation.apply` 还可能根据仪器交互结果覆盖显示位置，但这是显示层融合，未回写到共享后端地图；队友收到的是后端位置，不是浏览器融合后的屏幕点。本次不更改该既有显示功能。

## 3. 人物框和高清画面

代码路径：摄像头配置的 RTSP → FFmpeg `-c:v copy`（不裁剪、不缩放、不旋转、不镜像）→ 后端完整解码帧 → CameraWorker.resize（宽度大于960时等比例缩到960，高度取整）→ YOLO（框由 Ultralytics boxes.xyxy 返回输入图坐标）→ JPEG remote profile（再等比例缩到640）。

后端预览包含检测框/文字等绘制，非纯原始高清画面。YOLO 内部有预处理 letterbox，但导出的 boxes.xyxy 用于原输入帧，不应再按网络 letterbox 坐标解一次。应用代码没有 ROI 裁剪、黑边拼接、旋转或镜像操作。相机端电子裁剪/镜像或主辅码流视野差异并未现场验证，不能凭相同宽高比认定两条独立流一致。Cam1/Cam2 默认推流 Channels/102（辅码流），不是承诺 101 高清主码流。

当前 `frame_width/frame_height` 是交接 JPEG 尺寸；bbox 是检测输入尺寸。因此当前不能直接按 frame_width 转换 bbox 到高清帧。典型1920×1080源图：检测960×540，交接640×360；框应先乘 (640/960,360/540) 才与交接图一致，当前代码未做这一步。

建议在交接采集导出中记录真实检测尺寸和发送尺寸，然后只在交接模块缩放 bbox：x/w 按 Wsend/Wdet，y/h 按 Hsend/Hdet。接收端再用 Whd/Wsend、Hhd/Hsend换算。尺寸必须来自该帧元数据，不要仅假设960，也不要用 JPEG 宽高反推检测宽高。

如果摄像头另一路高清输入存在 crop/letterbox/rotation/mirror，必须提供经验证的变换矩阵或逐步变换参数；无法验证时禁止按比例套框。新变换字段须接收端 Schema 同步。本次没有改变当前图片发送或框语义。

## 4. 时间语义与原子性

当前 live 流程先 GET /api/state，再读取 /video/cam_2?remote=1 的 JPEG，两次读取并非同一帧快照。preview 线程取最新采集图，异步叠加最近检测结果；因此即使静止时看起来对齐，也不能证明移动/多人交叉时对齐。

当前 GPU 部署：原始 capture_received_at、地图 observed_at 来自 lnx 的 time.time()（接收RTSP帧时），不是主项目电脑的采集时间；snapshot_id 用本机发送时间生成，这不使 captured_at 自动变成本机权威时间。当前缺少把原始相机帧/PTS与主项目电脑采集时间关联的交接元数据。

应约定：captured_at 为被分析图像在主项目电脑采集入口的时间（若不具备相机曝光时刻则明确为host_receive）；occurred_at 为该snapshot.frame事件所描述的帧时刻，目前可继续等于captured_at；track.observed_at为该框实际检测帧时刻；location.observed_at为位置自身的真实观测时刻。输出时区转换不改变Unix时刻。不能拿发送时间、推理开始/结束时间覆盖采集时间。

本次交接层优先读取 position_observed_at；held/conflicted/predicted样本不给仪器位置候选；定位不可用时省略location，不填[0,0]。地图接口仍保留旧点年龄和估计状态。代码中“capture缺失时用now”的旧路径没有偷偷改为新契约，需要原子交接元数据接入后改为拒发/明确缺失。

最小后续方案为独立只读交接快照：一次导出 `{frame_id, camera_id, raw/source/detection/send尺寸, captured_at, bbox_observed_at, tracks, floor_map}`，必须绑定实际被分析的帧；现有检测结果可以缓存，但要保留检测时间并按双方容差拒绝过旧框。帧时间与主电脑时间的关联通过采集入口元数据完成；不硬编码时钟差、不改原始时间戳、不重构检测跟踪算法。此元数据现在不存在，不能声称已经保证同帧。

## 5. 结果回传与本次实现

此前 `_proxy_interaction_results` 直接透传，仪器位置显示层主要用 person_id/global aliases，未按 camera_id+snapshot_id+track_id 查找发送时人员；只有返回字段无法证明解析正确。

本次补齐内部 SnapshotTrackIndex：只记成功获回执的snapshot，最多300个；按完整三元键读取发送时该track的身份快照；相同track键重复、过期、缺少键或查不到均返回未知；不相信接收端返回的姓名，不按数组顺序分配。camera_id错误即使track_id相同也不能匹配。结果代理使用 `LAB_MEMBER_EXACT_RESULT_MATCH=1` 显式启用，默认关闭，现有运行进程未重启。启用后本地代理附handoff_match_status（不是发送给接收端的Schema字段），显示层对unmatched不做仪器位置融合，人物身份未知显示“未知人员”；matched仅用已解析global_id/person_id融合，不用裸track_id找人。旧结果不带此标记，原显示路径不变。索引只证明返回对应已发送元数据，不能修复原始元数据的帧错配。

身份读取补丁只从同相机 identities 按 local_id+global_track_id 唯一匹配已locked身份，保留未知null；未改变原身份算法。

仪器区域只能筛选需要分析的人/仪器，真假由队友MiniCPM输出。本次不调用或修改MiniCPM，不使用进入区域作为使用事件。

## 6. 地图版本与 Schema 建议

可选字段放在 snapshot.frame **顶层**：map_id、map_revision。不再增加平行的location_map_id等同义字段。location继续使用现有x_m/y_m/quality/observed_at。

`contracts/map-reference.schema-patch.proposed.json` 需合并到接收端实际snapshot.frame Schema：两字段是可选字符串，要求同时出现。若 Schema 版本不支持dependentRequired，使用等效成对presence规则；保留既有additionalProperties、Token、event_id幂等规则。是否升级schema_version由双方共同确定，不在本轮私自升级。

构造器增加include_map_reference=False默认参数。即便传入map_id/map_revision，默认也不会序列化。开启时两字段必须都存在，且来自同一份导出map_snapshot，不能额外请求另一版地图再套用。当前live publisher没有启用，未向运行接收端发送新字段。

## 7. synthetic示例与验证

运行 `python scripts/build_handoff_alignment_examples.py` 仅写本地JSON，不发网络事件：

- snapshot-frame.synthetic.legacy.json：当前字段结构与旧ID语义；即使向构造器传地图版本，也不发送。
- snapshot-frame.synthetic.proposed.json：提议的local track_id语义与顶层地图引用，需要双方同步后才可使用。

两个样例仅引用64×36无人物测试JPEG，bbox尺寸与该JPEG相符； synthetic位置/身份不是真实人。第二个track身份未知且省略location，不能被第一个人的位置填充。事件时间固定用于契约演示，不能作为现场实时测试。含person_id的示例仅是合成身份，实际身份发送应与既有数据分类政策一起核对。

已验证：位置老时间不变新、held省略、倒序地图列表仍按ID连接、歧义alias不匹配、身份绑定同local/global、完整三元键/缓存过期/未知名字处理，以及地图新字段默认不发送。现有member/frontend测试通过。未发送新synthetic到队友服务，未重新启动运行中的联调进程。

必须等待/共同验收：接收端正式Schema提供并同步地图字段；local/global ID语义切换；权威主电脑采集时间和同帧bbox/image绑定；双方原始高清码流变换核对；移动、遮挡、多人交叉的现场对齐。测试通过不代表多人不会错配。
