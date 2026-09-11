# 实验室人员—仪器交互模块移交包

## 当前版本

- 公共接口版本：`1.1`
- 插件实现版本：`0.2.0`
- 核心模型：MiniCPM-V 4.6
- 正式入口：`lab_instrument_interaction.InstrumentInteractionModule`
- 主要接入方式：24小时监控使用 `create_stream()` + `process_frame()`

## 接入方和接入方 AI 先做什么

1. 完整阅读根目录的 `模块接口使用说明.md`。
2. 不要直接调用 `模块/v4` 或 `模块/v5`，它们是可替换的内部实现。
3. 把依赖安装到真正启动接入方程序的 Python 环境。
4. 运行 `检查环境.py`，再运行 `检查环境.py --load-models`。
5. 参考 `模块/examples/realtime_stream_example.py` 接入实时摄像头。
6. 应用只创建一个 `InstrumentInteractionModule`，每个摄像头创建一个长期复用的流会话。

## 最小实时调用

```python
import sys
from pathlib import Path

PACKAGE_ROOT = Path(r"D:\仪器交互识别模块")
sys.path.insert(0, str(PACKAGE_ROOT / "模块"))

from lab_instrument_interaction import InstrumentInteractionModule

module = InstrumentInteractionModule(preload_models=True)
stream = module.create_stream(camera_id="lab_camera_view_2")

response = stream.process_frame(
    frame,                         # OpenCV BGR ndarray
    timestamp=elapsed_seconds,     # 非负且严格递增
    persons=recognition_results,   # 无人时传 []，缺结果时才传 None
    after_event_sequence=cursor,
)
cursor = response["last_event_sequence"]
```

退出时：

```python
stream.close()
module.close()
```

## 包内没有什么

- 没有复制开发电脑的 `.venv`，因为虚拟环境包含开发电脑的绝对路径，不能可靠迁移。
- 没有包含开发测试输出、缓存和演示录像。
- 不负责人员身份识别、摄像头断线重连、24小时录像保存、数据库或网页显示。

## 当前效果边界

高速离心机已有操作、路过/停留、看手机样本验证；真空干燥箱已有正样本验证。其余已标定仪器仍需要真实正负视频验证。接口已经具备正式接入条件，但未在接入方电脑完成双摄像头并发和24小时稳定性验证。

