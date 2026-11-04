# MiniCPM-V 4.6 本地部署

## 已部署内容

- 官方模型：`openbmb/MiniCPM-V-4.6`
- 本地模型：`D:\视觉模型\models\MiniCPM-V-4.6`
- 独立环境：`D:\视觉模型\.venv_vlm_next`
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU

## 已验证版本

- Python 3.11
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0 + CUDA 12.8
- Transformers 5.16.1
- PyAV 18.1.0

## 验证结果

- Safetensors 权重可完整读取，共 779 个张量。
- 模型可在 CUDA 上加载。
- 模型加载耗时约 3.2 秒。
- 模型静态显存约 2480 MB。
- 8 帧短视频测试峰值显存约 2812 MB。
- 单次 8 帧测试推理约 1.7 秒。

以上数字来自当前机器上的冒烟测试，不等于最终交互模块的实时性能。

## 测试命令

图像测试：

```powershell
& 'D:\视觉模型\.venv_vlm_next\Scripts\python.exe' `
  'D:\视觉模型\minicpm_v46_smoke_test.py' `
  'D:\素材视频\微信图片_20260819172309_168_7.jpg'
```

视频测试：

```powershell
& 'D:\视觉模型\.venv_vlm_next\Scripts\python.exe' `
  'D:\视觉模型\minicpm_v46_video_smoke_test.py' `
  'D:\输入视频\微信视频2026-08-20_161614_942.mp4'
```

视频由 PyAV 解码后均匀抽取 8 帧，再交给 MiniCPM-V 4.6。这样不依赖
`torchcodec`，也与后续固定仪器短窗口方案一致。

## 当前边界

模型已经部署并验证可以运行，但尚未接入正式的仪器交互模块。直接把整幅监控
画面交给模型会产生误判；下一阶段必须按照冻结方案加入目标仪器轮廓、目标人员、
短时间窗口和严格的有交互/无交互提示。
