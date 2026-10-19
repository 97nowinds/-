# 主项目视觉火焰检测：实现与实测

日期：2026-10-15。状态：推理引擎、录像评测和独立实时观察进程已跑通；候选模型与阈值尚未达到自动报警启用条件，生产监控未重启。

## 已实现

- fire_detector.py：ONNX Runtime CPU，2 个推理线程，固定权重摘要校验；从模型元数据读取 fire/smoke 类别，不能把 COCO 人员模型当作火焰模型。RGB 归一化、letterbox、逐类别 NMS，框还原到原始画面。
- fire_worker.py：共享现有只读冻结快照，不新增 RTSP 连接；每轮取最新帧、不积压。进程文件锁防止重复启动；网络错误只留异常类型，不输出 Token 或 RTSP 信息。
- scripts/evaluate_fire_detector.py：仅使用录像文件，不接通知服务。
- config/fire_model.json 与 scripts/download_fire_model.py：记录固定版本及源/导出 SHA256，先校验 checkpoint，再限制为明确的已安装 Torch/Ultralytics 类型、weights_only=True 加载并导出 ONNX；运行时只加载 ONNX。拒绝未知类型，导出摘要不一致则停止。模型文件忽略 Git。已实际运行下载与导出脚本，产物摘要一致。
- 本机没有 CUDA，实际使用 CPU；无需为当前观察任务抢占人员追踪 GPU。

## 实际评测

基础候选：[fiacecson20/cctv-ai-fire-smoke](https://huggingface.co/fiacecson20/cctv-ai-fire-smoke)，固定 revision=343990e42d99a5d27e9f35fc7c80880dc5f43f45。输入 320×320；元数据实际输出为 [1,6,2100]，不是照抄模型卡中的通用输出示例。下载 SHA256 与上游 LFS 摘要一致。模型卡的作者许可声明与导出元数据有差异，当前配置保留导出元数据的 AGPL-3.0 标记，未将其标为无条件商业可用。

使用 camera/runtime/yxq_side_20261009 中三路录像，各均匀抽查 40 帧：

| 录像 | 抽查帧数 | 火焰候选帧（分数≥0.25） | 可报警帧（≥0.9） | 推理中位数 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cam_1 | 40 | 0 | 0 | 17.18 ms | 23.22 ms |
| cam_2 | 40 | 0 | 0 | 15.95 ms | 21.38 ms |
| cam_entrance | 40 | 3 | 0 | 15.53 ms | 19.76 ms |

这些是正常实验室录像，未作为逐帧独立人工标注测试集，不能据此报告准确率、召回率或整个录像零误报。入口 3 帧低分候选应保留为干扰样本。

独立进程真实读取三路快照约 35 秒：54 帧、0 个读取/推理错误、0 次结果提交、0 次群通知。末轮单帧推理约 16–21 ms。观察结束进程已退出；原 15001、5173、5175 服务均未因本轮实验而重启。不能据此声称所有原服务的端到端延迟不变。

本机报告：

- runtime/fire-offline-evaluation-20261015.json
- runtime/fire-worker-status.json

## 正样本检查发现的问题

对作者提供的带标注验证拼图做引擎冒烟检查，整图最高 fire 分数约 0.53，分块后最高约 0.785。对 [D-Fire 作者公布的示例图](https://github.com/gaia-solutions-on-demand/DFireDataset) 分块检查，明显火焰示例最高约 0.77，部分火焰画面没有 ≥0.25 的候选框。

这些带标注示例不是独立、无污染的测试集，不能计算真实召回率；但已经足以说明，不能凭正常录像不报警就批准模型。

还比较了两个固定版本的公开候选：

| 候选 | 实测 | 结论 |
| --- | --- | --- |
| mfranzon/fire-smoke-yolov8，f1c6426b069c1849cbf13b1ef5d2a260289286db，导出 320 | 同一公开火焰示例最高约 0.656；本地 120 帧无报警 | 仍未达到 0.9 的启用条件 |
| rabahdev/fire-smoke-yolov8n，13017fe8af477c25f5298d168e2dfede4b000753，导出 640 | 同一示例最高约 0.693；本地 120 帧无报警，中位推理约 59–61 ms | 未证明更适合当前场景，计算开销增加 |

后两个候选只在离线阶段加载：检查 checkpoint 全局引用，仅允许已安装 Torch/Ultralytics 的明确类型，使用 weights_only=True，不使用远端 Python 代码。正式推理工作进程仅使用 ONNX 文件。比较报告位于 runtime/fire-candidate2-comparison.json 和 runtime/fire-candidate3-comparison.json；这些候选未替换当前服务模型。

## 原始标注的小样本评测与候选选择

又从公开 D-Fire test 镜像读取了原图和原始标签。镜像为 hanvithSai/gavin-dfire-raw，固定 revision=fee27ef6934d5bb569ae6271c123673f10a877c8，原作者及许可按 D-Fire / CC BY 4.0 保留。标签约定 0=smoke、1=fire。按有限值、中心范围及正尺寸检查排除 4 份异常标注，剩余 4302 份标签中，按固定 hash 顺序抽取火焰、仅烟雾、无烟火各 20 张，总计 60 张。超出边界的框在计算 IoU 时裁到图像边界。

| 模型 | 阈值 | 20 张火焰图中检出候选 | 其中框与真值 IoU≥0.5 的图数 | 40 张非火焰图误报图数 |
| --- | ---: | ---: | ---: | ---: |
| 基础候选 320 | 0.5 | 5 | 4 | 0 |
| mfranzon 320 | 0.5 | 2 | 2 | 0 |
| rabahdev 640 | 0.25 | 17 | 14 | 0 |
| rabahdev 640 | 0.5 | 14 | 12 | 0 |
| rabahdev 640 | 0.9 | 1 | 1 | 0 |

第三个候选在这批样本上更适合作为下一阶段基线，config/fire_model.json 已改为该候选的独立观察配置，路径 models/fire/rabahdev-yolov8n-640.onnx。状态仍是 experimental_observe，没有替换正在运行的人员模型，也没有批准自动群通知。原基础候选 ONNX 与评测记录保留。

不能将 0/40 误报外推为真实实验室零误报，也不能把含火图有候选等同于所有火焰框都被检出；这是小样本帧级统计，模型与公开数据训练来源可能相关。当前 0.9 阈值明显漏检；需要更完整的正负验证和连续视频评估，再确定阈值与持续时间。没有直接修改生产阈值。

数据及结果：runtime/fire-benchmark/manifest.json（图像摘要与标签）、runtime/fire-benchmark-baseline.json、runtime/fire-benchmark-candidate2.json、runtime/fire-benchmark-candidate3.json。评测代码 scripts/evaluate_fire_benchmark.py 不调用通知或实时接口。

第三个候选另做了 30 秒实时只读观察：45 帧、0 错误、0 提交、0 群通知；观察进程已正常退出。先前基线观察状态另存为 runtime/fire-worker-baseline-status.json，新的状态保存在 runtime/fire-worker-status.json。

## 通知与恢复的门槛

1. config/fire_model.json 的 state 当前为 experimental_observe；工作进程拒绝 submit。
2. 即使模型获准，也必须有有效的 source→当前主机 authority 时钟映射、相同源帧和 JPEG 摘要。当前读取的三路冻结快照无有效 sync_mapping；不改写 captured_at 为发送时刻来绕过。
3. 新的场景事件使用独立 fire: 快照键，不覆盖人员交接的历史身份绑定，也不修改队友 snapshot.frame Schema。
4. 没有火焰框不意味着火焰已熄灭，当前不自动输出 clear。恢复策略需独立验收。
5. 用户已提供传感器安装点的地图红框；它不是覆盖范围。未启用整间实验室或整幅相机画面的合并。

## 复现

工程回归共 76 项通过，包含类别编号反转、原图坐标还原、权重摘要、拒绝未知 checkpoint 类型、时钟域与有效期校验，以及原有融合、图文通知、鉴权和跟踪交接测试。单元测试通过不等于视觉模型准确率通过。

在仓库根目录，用现有 Python 环境执行：

```powershell
.\camera\.venv\Scripts\python.exe camera/scripts/download_fire_model.py
.\camera\.venv\Scripts\python.exe camera/scripts/evaluate_fire_detector.py camera/runtime/yxq_side_20261009/cam_1.mp4 camera/runtime/yxq_side_20261009/cam_2.mp4 camera/runtime/yxq_side_20261009/cam_entrance.mp4 --samples 40 --output camera/runtime/fire-offline-evaluation-20261015.json
.\camera\.venv\Scripts\python.exe camera/fire_worker.py --seconds 60 --interval 2
```

最后一条默认只观察，没有通知副作用。视觉模块当前需要补充无标注覆盖框的独立正负样本集、对难例做阈值与时序验证，必要时做场景微调；不能仅靠降低阈值获得“报警成功”。硬件报警也不应依赖视觉模块通过验证后才发送。
