# Codex 接入约束

你正在接入一个可替换的实验室人员—仪器交互插件。

开始修改接入方项目之前，必须完整阅读 `模块接口使用说明.md` 和 `模块/examples/realtime_stream_example.py`。

- 公共入口只有 `lab_instrument_interaction.InstrumentInteractionModule`。
- 不要从 `v2`、`v4`、`v5` 导入私有函数，也不要把内部实现复制到接入方代码。
- 24小时监控使用 `create_stream()` 和 `process_frame()`；`analyze_video()` 只用于离线录像。
- 应用进程只创建一个模块对象；每个摄像头只创建一个长期会话。
- 每帧提供接入方人员识别列表，无人时传 `[]`，只有识别结果缺失时才传 `None`。
- 人员框必须是原始输入帧像素坐标，`person_id` 跨帧保持稳定。
- `process_frame()` 非阻塞，其输出是后台最新状态，不是同步逐帧结论。
- 消费并保存 `events` 和事件游标；监控 `error`、`event_gap_detected`、`analysis_lag_seconds`、`dropped_frames`。
- 不要把错误或未完成分析解释成“无交互”。
- 停止或升级前先关闭所有流，再关闭模块。
- 未经回归测试，不要修改提示词、仪器轮廓、阈值和融合规则。

目标是通过稳定公共接口完成接入，让以后替换整个插件目录时不必改动接入方业务代码。

