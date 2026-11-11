# 11月11日版本

大屏通过前端服务转发实时数据、视频和集群 3D 帧。集群 3D 服务通过
camera/scripts/cluster_twin_tunnel.ps1 提供本机端口 5182。地图位置快照
equipment_positions.json 对应部署中的 camera/runtime/equipment_positions.json。

wizard/ 保存 LabSuite_FinalRoute_20261109 向导的集群执行入口。
inference_server.py 与 cluster_inference.py 放在向导原目录。
同目录的 cluster_connection.txt 只在部署机器配置，内容是本地
resident_services.json 的完整路径。wizard_live_worker.py 放在集群
/cluster/dataX/xqyi/labsuite-20261110/，使用该目录中已有的
wizard-revision27-final 模型环境和预置画面。
