# 飞书群通知接入

更新：2026-10-15。当前使用群 Webhook，接收人员直接加入对应群，不需要注册额外网站。

2026-10-19：新版通知端已加载到 5173，三群分级路由与 L0 静默规则已核对生效。检测侧尚未接入，详见 [当前部署与验收记录](FEISHU_NOTIFICATIONS_20261019.md)。

同日补充：entrance 首次确认身份后提供 120 秒穿戴准备期；学生群 → 负责师兄师姐群分级处理普通防护问题，老师群接收火焰等严重事件。没有固定准备区。详见 [准备期、传输验证与已知边界](PPE_PREPARATION_AND_GROUP_ROUTING_20261019.md)。

具体路由、证据要求、告警规则与部署状态见 [群通知机制](MANAGEMENT_MECHANISMS_20261015.md)。

三个群及图片上传使用 scripts/configure_feishu_evidence.py 配置；单群连接工具为 scripts/configure_feishu_group.py。凭据在本机隐藏输入并用 DPAPI 加密保存，不写入 Git。旧私聊、联系人查询、电话及个人网站账号功能已移除。

后台只保留维护管理员与检测服务 Token。维护页 /management 为可选配置工具，普通接收人员不需要访问。日常进入与正常交互只记录；防护违规附快照发实验人员群；持续违规升级到负责人群；火焰告警发三个群。

## 给队友的口罩结果契约

独立本机管理接口，不修改现有 snapshot.frame 的成员 Schema；双方确认后接入。口罩检测仍由队友模型完成。本轮不会主动向队友发送任何消息。

必须回传 **(camera_id, snapshot_id, track_id)**，使用原始快照编号；身份由主项目索引恢复。person_id、display_name等回传值不会用于指派姓名。

以下为**合成数据**，不可用于真人事件或现场验收。时间必须换成对应真实快照的时间，不能替换成发送时间：

```json
{
  "evidence_type": "mask",
  "analysis_id": "synthetic-mask-001",
  "camera_id": "cam_2",
  "snapshot_id": "synthetic-snapshot-001",
  "track_id": "synthetic-track-001",
  "time_domain": "authority",
  "captured_at": "2026-10-12T03:00:00.000+00:00",
  "published_at": "2026-10-12T03:00:01.000+00:00",
  "expires_at": "2026-10-12T03:00:06.000+00:00",
  "status": "valid",
  "mask_state": "not_worn",
  "confidence": 0.95
}
```

mask_state 可使用 worn/not_worn/unknown；只有 not_worn 且有效才产生事件。status 非 valid 代表无可用判断，不应伪造 not_worn。expires_at 最多有效到 published_at+5秒。主项目快照发布进程必须持续登记原始快照，通知模块从发布器或本机共享索引读取对应记录；无索引、不匹配、时间变化、身份未知均不接收为实名告警。提交成功但 `accepted=false` 表示无新可通知事件，也可能是冷却/重复，并不表示检测人物戴了口罩。

仪器侧自动轮询已存在的 `lab-interaction-results-1.2` analyses；若采用主动提交，evidence_type=interaction，并提供 instrument_id、可选 instrument_name、is_interacting 和同样的时间/快照字段。instrument_id 必须用队友现有仪器配置的真实编号；不会硬编码“超声波净化器”的ID。

## 现场验收与回退

本轮验证：通知模块15项测试通过（含真实本地 HTTP 页面/401认证检查）；交接快照关联5项、联合演示11项回归通过。原有前端分离测试42项中41项通过，1项既有阈值断言失败：测试要求0.95，而 Git HEAD 地图配置已是0.75，本轮未改该配置。为运行已声明的Schema回归检查，补装了本地 `.venv` 缺失的 `jsonschema>=4.18,<5`，未更改requirements约束。`py_compile`与`git diff --check`通过。

1. 单人入口进入室内：只留档，不发群消息；仅路过入口、离开、回头需验证不误记进入。
2. 两人交叉：每条消息的人员身份和原始快照框人工核对；未知人员不套用 yxq。
3. 仪器：队友 true结果触发，地图候选/false/pending不触发；过期/错快照/改时间拒绝。
4. 口罩：队友真实结果 not_worn≥0.9触发；unknown、遮挡、低分不触发。
5. 技术鉴权：无Token=401，越权=403；producer只能提交指定范围检测结果，维护管理员查询和配置。实验人员、负责人和管理人员直接按飞书群成员接收，不另外注册账号。
6. 飞书：机器人配置后，在授权管理群实测送达和状态；断网故障隔离、恢复重试、过期不补发。当前所有发送测试使用假发送器，未发送真实飞书消息。
7. 回退：去掉 `-EnableManagement`，设置 LAB_MANAGEMENT_ENABLED=0 后重启前端；不删除事件数据库、不改跟踪模型、不改队友交接Schema。
