"""24小时监控接入示例。实际使用时把示例人员结果换成队友算法输出。"""

import sys
import time
from pathlib import Path

import cv2


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from lab_instrument_interaction import InstrumentInteractionModule


def teammate_recognize_people(frame):
    """这里替换成队友已有的人员识别和跟踪功能。"""
    return [
        {
            "person_id": "employee_001",
            "person_name": "张三",
            "bbox": [281, 1, 407, 428],
            "confidence": 0.96,
        }
    ]


module = InstrumentInteractionModule(preload_models=True)
capture = cv2.VideoCapture(0)  # 也可以替换成 RTSP 地址
if not capture.isOpened():
    raise RuntimeError("无法打开摄像头")

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
stream = module.create_stream(
    camera_id="lab_camera_view_2",
    frame_size=(width, height),
)

start_time = time.monotonic()
event_cursor = 0

try:
    while True:
        ok, frame = capture.read()
        if not ok:
            # 实际系统应在这里执行摄像头断线重连，而不是报告“无交互”。
            time.sleep(0.2)
            continue

        timestamp = time.monotonic() - start_time
        persons = teammate_recognize_people(frame)
        response = stream.process_frame(
            frame,
            timestamp=timestamp,
            persons=persons,
            after_event_sequence=event_cursor,
        )
        event_cursor = response["last_event_sequence"]

        if response["error"]:
            raise RuntimeError(response["error"])

        # current_interactions 用于刷新网页当前状态。
        current_interactions = response["current_interactions"]

        # events 用于保存开始/结束记录或推送消息。
        for event in response["events"]:
            if event["type"] == "interaction_started":
                print(
                    "开始交互：",
                    event["person_name"] or event["person_id"],
                    "→",
                    event["instrument_name"],
                )
            elif event["type"] == "interaction_ended":
                print(
                    "结束交互：",
                    event["person_name"] or event["person_id"],
                    "→",
                    event["instrument_name"],
                )

        # 可选：监控是否跟不上。大于0表示输入帧曾因队列满被丢弃。
        if response["metrics"]["dropped_frames"]:
            print("警告：交互模块输入队列出现丢帧", response["metrics"])
finally:
    capture.release()
    stream.close()
    module.close()
