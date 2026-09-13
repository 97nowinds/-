# HAFS 硬件接入说明

## 已识别的硬件功能

| 功能 | STM32 引脚/接口 | 当前行为 |
| --- | --- | --- |
| DHT 温湿度 | PA1，单总线 | 每秒采集，成功时更新 OLED |
| 火焰继电器触点 | PB1，内部上拉，低电平告警 | 告警时蜂鸣器和 LED 常亮 |
| 蜂鸣器 | PB0，高电平有效 | 跟随火焰告警 |
| 板载 LED | PC13，低电平点亮 | 跟随火焰告警 |
| SSD1306 OLED | I2C1，PB6/PB7，地址 0x3C | 显示温度、湿度与 FIRE 状态 |
| 电脑通信 | USART1，PA9 TX / PA10 RX | 115200、8N1，无流控 |

当前固件只主动发送文本诊断行：

```text
RAW: 02 26 01 08 31
[SAFE] System Normal
```

或：

```text
[ALARM] FIRE DETECTED!
```

电脑侧兼容层 `hardware_serial.py` 可以解析上述格式，也接受推荐的 JSON Lines：

```json
{"device_id":"ENV01","sequence":1024,"temperature":26.4,"humidity":51.2,"flame_alarm":false,"dht_valid":true}
```

## 启动串口接入

先安装依赖，然后启动后端。后端会自动识别 CP2102/CP210x 和 CH340/CH341；本次实机的 CP2102 会自动选中 COM3：

```powershell
python -m pip install -r requirements.txt
python app.py
```

需要指定端口时可设置 `LAB_HARDWARE_PORT=COM3`，它的优先级高于自动识别；设置 `LAB_HARDWARE_AUTO=0` 可关闭自动识别。波特率默认 115200，可用 `LAB_HARDWARE_BAUD` 修改。未发现支持的串口时，后端和页面仍可正常启动，硬件区域显示“未配置”。默认超过 30 秒没有有效数据即显示“数据超时”；可用 `LAB_HARDWARE_STALE_SECONDS` 调整。

当页面通过 `frontend_server.py` 连接远端视觉后端时，串口由本机前端服务读取，并通过同源的 `/api/environment` 合并到页面状态。这样 COM3 不需要也不能转发给远端服务器，现有摄像头与识别链路保持不变。

也可以由独立串口进程或测试工具推送标准 JSON：

```http
POST /api/environment/ingest
Content-Type: application/json

{"device_id":"ENV01","temperature":26.4,"humidity":51.2,"flame_alarm":false,"dht_valid":true}
```

## 上板前需要修正/确认

1. `dht11.c` 当前按“16 位数值除以 10”解析，这是 DHT22 常见格式；标准 DHT11 通常按整数字节和小数字节解析。应按实物型号修正固件，否则读数可能严重失真。
2. `HAFS.ioc` 未声明 PB1，但 `BSP_FlameRelay_Init()` 会在运行时配置 PB1。后续重新用 CubeMX 生成代码前，应把 PB1 输入配置写回 `.ioc`，避免配置丢失。
3. 当前 UART 没有稳定的单条完整状态报文，`RAW` 与火焰状态分成两行。建议固件最终改为每秒发送一条 JSON Lines，并加入 `sequence`，便于检测丢包和去重。
4. 火焰、烟雾等安全设备必须保留独立的现场报警回路；网页和 STM32 状态只适合作为辅助监测，不能代替认证的联锁、急停或安全继电器。
