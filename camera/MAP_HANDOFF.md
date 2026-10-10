# 平面图与人员位置交接

当前专用只读共享服务：`http://192.168.31.127:5175`。

- `GET /api/handoff/map`：地图布局和当前人员位置。
- `GET /api/handoff/positions`：人员位置和三路摄像头轨迹。
- `GET /health`：服务和摄像头后端连通状态。

推荐接收端服务每秒读取一次，不需要接收端 Token。接口只读，适用于实验室局域网。现有 5174 联合前端下次重启后也提供前两个接口。

`layout` 包含地图宽高（米）、实验台/门等 fixtures、通道 zones 和摄像头位置/FOV。区域可能包含 points 多边形，接收端应优先采用 points，其他区域按 x/y/width/height 绘制。坐标原点为左上角，x 向右、y 向下。

`map_id` 为地图身份，`map_revision` 为布局版本；人员移动不会改变布局版本。

`people` 是当前可显示的地图人员，坐标为 `x_m`、`y_m`。保留 `track_id`、身份、来源相机、`observed_at`、`position_observed_at`、`position_age_seconds`、`position_estimated`、`localization_state`、`count_status` 等原始字段。用 track_id 与画面事件中的 global_id 对应；注册身份按实际提供字段读取，不要用相机 local_id 作为全局身份。

`camera_tracks` 提供包括无法地图定位的检测轨迹，以及重复计数关系。没有地图点不等于没有检测到人。`people: []` 表示当前没有地图人员；HTTP 503 表示后端不可用，应显示断连，不能解释为实验室无人。

数据每次请求取最新后端状态，界面仍可每 5 秒更新一次打点。估计/保持位置应标识并显示年龄，不作为当前精确测量。

当前地图使用近似标定（calibration_status），仪器确认位置需要与此 map_id/map_revision 和米制坐标对应，再按置信度融合。

示例：

```python
import requests
snapshot = requests.get('http://192.168.31.127:5175/api/handoff/map', timeout=5)
snapshot.raise_for_status()
data = snapshot.json()
layout = data['layout']
for person in data['people']:
    print(person['track_id'], person['x_m'], person['y_m'], person.get('position_estimated'))
```
