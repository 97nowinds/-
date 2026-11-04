# MiniCPM-V 4.6 实时优化版

## 运行

把 MP4 视频拖到 `运行MiniCPM实时优化版.bat` 上。结果生成在：

`outputs\minicpm_v46_realtime\视频名_运行时间`

其中：

- `annotated.mp4`：最终交互状态视频；
- `result.json`：MiniCPM调用记录、跳过记录、交互时间段和耗时；
- `review_inputs`：实际提交给MiniCPM的连续画面。

准确版入口 `运行MiniCPM新方案.bat` 仍然保留，方便以后进行准确率对照。

## 实时优化原理

每秒仍然形成一个包含最近2秒、8帧画面的候选窗口，但不再把33个窗口全部提交给MiniCPM。

1. 固定仪器区域周围没有检测到人员时，跳过MiniCPM。
2. 人员存在时，计算目标仪器附近的画面活动量。该活动量只决定是否调用MiniCPM，不直接决定有无交互。
3. 活动开始、活动明显增强或持续较长时间时，连续复核相邻窗口，避免关键动作刚好位于两个审核间隔之间。
4. MiniCPM首次判真后进入短时逐秒复核。
5. 最近三个窗口至少两个由MiniCPM判真，才正式进入“正在交互”；进入后，下一次MiniCPM明确判假时退出。

最终的 `true/false` 仍来自MiniCPM-V 4.6。活动量、人员位置和仪器区域不代替MiniCPM作交互结论。

## 当前明确未使用的内容

- 不使用Qwen视觉模型；
- 不使用旧版手腕—仪器距离规则；
- 不使用姿态关键点判断交互；
- 不识别具体操作类别。

YOLO普通人员检测模型只用于确认目标仪器附近是否有人，以及给MiniCPM指定目标人员。

## 固定参数

- MiniCPM视觉压缩：4×；
- 时间窗口：2秒；
- 窗口检查步长：1秒；
- 每个MiniCPM输入：8帧；
- 空闲周期复核：6秒；
- 活动触发阈值：0.04；
- 正向置信度要求：不低于0.75；
- 前置提示词：`prompts\minicpm_v46_interaction_system_prompt.txt`。

提示词仍是外部文本文件，修改后下次启动自动生效。

## 三段基准测试

| 视频内容 | 时长 | 实际耗时 | MiniCPM调用 | 总候选窗口 | 结果 |
|---|---:|---:|---:|---:|---:|
| 操作高速离心机 | 33.08秒 | 36.34秒 | 15 | 33 | true，2段交互 |
| 站立/停留未操作 | 5.92秒 | 6.26秒 | 1 | 5 | false |
| 仪器前看手机未操作 | 9.84秒 | 9.79秒 | 3 | 9 | false |

准确版对33秒正样本需要约70秒、调用33次。实时优化版将调用降到15次，同时保留两个正确交互时间段。

## 常驻模型与两种调度的对照测试

测试时MiniCPM和人员检测模型各加载一次，然后连续处理操作正样本、站立负样本、看手机负样本和同摄像头无人片段。四段总时长54.8秒。

| 调度方式 | 四项判断 | MiniCPM总调用 | 不含首次加载的处理时间 | 实时系数 |
|---|---:|---:|---:|---:|
| 自适应（默认） | 全部正确 | 19 | 42.12秒 | 0.77 |
| 普通2秒、活动1秒 | 全部正确 | 36 | 59.98秒 | 1.09 |

实时系数小于1表示处理速度快于画面产生速度。因此默认保留自适应调度；严格的普通2秒/活动1秒调度已经实现，可通过命令行参数 `--schedule-mode normal2_active1` 进行对照，但不作为默认设置。

无人片段共有5个候选窗口，人员检测均为0，MiniCPM调用次数为0。

完整对照结果：

- `outputs\minicpm_v46_persistent_benchmark\adaptive\benchmark.json`
- `outputs\minicpm_v46_persistent_benchmark\normal2_active1\benchmark.json`

## Python常驻调用入口

队友的监控进程启动时只调用一次：

```python
from v5.minicpm_realtime_video import create_persistent_resources

judge, person_detector = create_persistent_resources()
```

后续处理连续视频片段时复用这两个对象：

```python
from v5.minicpm_realtime_video import analyze_video_realtime

result = analyze_video_realtime(
    video_path,
    judge_instance=judge,
    person_detector_instance=person_detector,
)
```

不能在监控循环中反复调用 `create_persistent_resources()`，否则会重复加载模型。

## “实时”的范围

当前BAT是用已有视频验证实时处理速度。一次启动包含约3秒模型加载和结束后的结果视频转码，因此总时间略高于视频时长。

接入持续监控时，应让MiniCPM实例常驻显存，并不断把最新2秒画面送入本模块；不能每两秒重新启动一次Python程序。常驻后不再重复模型加载，画面采集也不需要输出视频转码。

当前只在三段素材和一个固定摄像头视角上验收。接入真实监控前仍需增加更长时间、更多人员、遮挡和光照变化样本。
