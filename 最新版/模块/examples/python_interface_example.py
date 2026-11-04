"""队友程序调用示例：请只从 lab_instrument_interaction 导入公共入口。"""

import sys
from pathlib import Path


# 如果“模块”目录没有安装到 Python 环境，就把它加入搜索路径。
MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from lab_instrument_interaction import InstrumentInteractionModule


identity_timeline = [
    {
        "timestamp": 0.0,
        "persons": [
            {
                "person_id": "employee_001",
                "person_name": "张三",
                "bbox": [281, 1, 407, 428],
                "confidence": 0.96,
            }
        ],
    },
    {
        "timestamp": 0.2,
        "persons": [
            {
                "person_id": "employee_001",
                "person_name": "张三",
                "bbox": [280, 1, 418, 428],
                "confidence": 0.97,
            }
        ],
    },
]


# 应在应用启动时创建一次，并在多个视频之间复用。
module = InstrumentInteractionModule(
    default_camera_id="lab_camera_view_2",
    preload_models=True,
)

try:
    print(module.health_check(require_loaded=True))
    result = module.analyze_video(
        r"D:\待分析视频\camera_2_clip.mp4",
        camera_id="lab_camera_view_2",
        identity_timeline=identity_timeline,
    )
    for event in result["interactions"]:
        print(
            event["person_id"],
            event.get("person_name"),
            event["instrument_id"],
            event["instrument_name"],
            event["start_time"],
            event["end_time"],
        )
finally:
    # 应用退出或替换插件前调用。
    module.close()
