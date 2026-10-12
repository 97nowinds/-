# 入口相机与地图恢复检查（2026-10-12）

## 已确认的原因

1. 实际运行的集群监控页调用了 `InstrumentLocation.apply`，但未部署对应静态脚本，也没有在模板加载。浏览器重复抛出 `ReferenceError: InstrumentLocation is not defined`，刷新在地图之前中断；后端仍有位置输出。
2. 本机 `scripts/restore_hd_push.py` 当前只发布 cam_1、cam_2。读取本机受保护配置的字段元数据确认仅有这两路高清源，没有入口源；两路FFmpeg输出分别到 `127.0.0.1:18554/cam_1`、`/cam_2`。集群实时进程等待 `/cam_entrance`，因此入口永远重连。本机入口IP 192.168.31.192 的554端口可达，但TCP可达不证明RTSP认证和画面正常。
3. Windows PowerShell 5.1 使用系统编码读取无BOM的UTF-8脚本，中文字符串变成乱码，并导致引号解析失败。这发生在启动通知服务之前。

## 已完成修复及验证

- 本地 `static/app.js` 添加可选仪器定位模块缺失时回退到原始 floor_map 的保护，避免整个地图停止刷新。
- 向当前实时集群部署对应脚本、补充模板加载，运行中的JS加入同样回退保护；原JS备份到集群 `camera/runtime/backups/app-map-*.js`。未重启GPU检测进程，未修改地图标定、跟踪或身份算法。
- 浏览器刷新后显示“服务在线”，实际出现过一名副通道未注册目标；地图仍按照5秒采样展示，不保证每时刻都有人。对应脚本HTTP200，地图是原有示意映射，标定质量没有被改成正式标定。
- 本地启动脚本写为UTF-8 BOM，使用Windows PowerShell原生Parser验证通过。

## 入口恢复与当前待办

新增 `scripts/restore_entrance.py`，本机隐藏输入入口RTSP账号密码，先验证摄像头画面，再只恢复入口推流到 `/cam_entrance`。本地Windows当前用户DPAPI加密保存到 `runtime/entrance-publisher.dpapi`。FFmpeg输出不记录凭据；独立守护程序监控连接并在断开后重试。

用户完成凭据输入后已验证：入口本机推流状态 publisher_connected；集群 cam_entrance=running，采集约20FPS、画面年龄约0.1秒；监控页面显示真实入口画面及3/3路正常。入口人脸检测已经运行（检查时33次尝试、24次检测），但匹配次数仍为0、identity为空，因此不能声称已恢复yxq实名识别。恢复截图保存为 runtime/entrance-restored-20261012.png。

飞书通知前端仍需用户重跑修复后的原启动命令，本轮没有再尝试被自动审批拦截的通知启动操作。此前群连接测试成功不等于全部事件监听已经运行。
