# 人员事件、飞书通知与管理权限（2026-10-12）

## 当前采用群通知（取代点对点接入）

根据用户最新要求，当前默认及已有持久化路由切换为 laboratory_group，仅通过群自定义机器人 Webhook 发送。点对点适配保留为代码能力，不在当前路由使用；电话未启用。

本机运行 `scripts/configure_feishu_group.py`，填写目标群 Webhook 和可选签名密钥。测试按钮会向此群发送明确标注的测试消息。成功后凭证由 DPAPI 加密保存为 `runtime/feishu-group.dpapi`，优先加载群配置。群自定义机器人不要求手机号/邮箱查询或应用用户编号权限。若机器人有关键词限制，请允许“实验室”；如果有IP白名单，请按实际出口地址配置。服务需在当前 Windows 用户下启用通知管理，密钥不写入Git。

## 飞书应用机器人点对点接入补充

新增 `feishu_app.py`：获取并缓存 tenant_access_token，按手机号/邮箱查询此应用下的 open_id，发送私聊消息；每个 event_id+接收人生成稳定请求 uuid。群 Webhook 兼容保留，路由 transport=app/recipient_key=primary 使用本机配置的测试接收人。当前仅支持一个 primary 接收人，多负责人路由待后续配置。

运行 `scripts/configure_feishu.py` 打开本机输入窗口；App Secret、手机号/邮箱输入隐藏。按钮明确授权向填写的接收人发送一条连接测试消息；只有查询接收人并获得发送成功响应后才保存加密凭证，并将通知路由切换为该接收人的私聊。返回成功代表飞书接受发送请求，还需接收人确认看到消息。

应用凭证、接收人 open_id 和随机管理令牌保存在 Windows 当前用户 DPAPI 加密的 `runtime/feishu-app.dpapi`，不写明文到配置或日志。这个文件仅同一 Windows 用户可解密，迁移主机应重新配置。管理运行时自动加载它；前端启动仍需启用通知管理。电话加急接口已封装，但正常通知队列没有电话调用，配置保存 phone_enabled=false，未启用严重事件自动升级。

本轮接入尝试停在接收人查询：应用验证后未找到唯一有效 open_id，尚未发送消息、尚未保存正式凭证。需核对组织通讯录账号和应用通讯录数据范围，发布修改后在原窗口重试。新增应用客户端6项测试（假API）及通知管理15项测试通过，DPAPI本机加解密验证通过；不能据此声称真实飞书接入完成。

## 当前交付与启用条件

新增模块不修改检测、ByteTrack、ReID、人脸身份和地图算法，也不生成仪器交互判断。`frontend_server.py` 加载通知管理模块，默认关闭；启动脚本支持 `-EnableManagement`。未配置飞书机器人时不发送消息。现有运行进程尚未加载本轮代码，未重启它们。

管理页面：**前端实际地址 + `/management`**。不是 GPU 后端的 15001 端口。管理 API 与普通监控前端共用进程；通知采集和飞书发送各有独立线程。关闭浏览器不停止通知采集。

启用前在本机设置 `LAB_MANAGEMENT_ADMIN_TOKEN`，再以原先参数运行 `scripts/start_frontend.ps1 -EnableManagement`。保持现有后端地址和成员交接参数；不要覆盖现有联调配置。令牌使用随机高强度值，不使用默认密码。远程访问通过可信内网/Tailscale或 HTTPS 反向代理，不将 Bearer 令牌通过公网明文 HTTP 传送。

配置文件：`config/management.json`；事件、投递状态、路由更改和审计保存在忽略 Git 的 `runtime/management.sqlite3`。队列重启后保留；最多重试 5 次。事件默认 15 秒内有效，投递超过观测时间 300 秒后标记 expired，不会在几天后补发过期告警。默认同类人员/仪器通知冷却 120 秒，避免摄像头切换和重复轮询刷屏。

## 事件依据与限制

|事件|触发条件|不能当作证据的内容|
|---|---|---|
|person.entered|实时入口摄像头已锁定身份，20秒内室内摄像头出现同一已锁定 person_id；两处都对应实际活跃框|首次在室内看到人、录像回放、过期检测、仅姓名|
|instrument.started|成员结果 status=valid、is_interacting=true；完整快照键匹配发送时身份；权威时钟域、时间有效|地图点进入仪器区域、false、pending、身份未知|
|safety.no_mask|成员提交 mask_state=not_worn、confidence≥0.9，并满足同样快照/身份/时间校验|脸部看不清、unknown、低置信度、检测缺失|

入口通知是**入口后室内的身份连续观测规则**，不是门线方向检测。它尚未经过现场进出方向验收：若入口与室内视角存在重叠、离开前回头等情况，仍可能误判方向。通知文字保留证据说明。已在室内持续出现的人员不会因再次被入口相机看到而触发；室内缺席60秒后解除此抑制。遮挡超过60秒不能据此证明已离开；本轮不提供精确进出计数或占用人数。

进入事件 observed_at 是**主项目收到室内确认的时刻**，evidence_source=`host_received_entrance_then_indoor`，不是宣称摄像头采集时刻；不会修改源检测时间。交互/口罩事件 observed_at 使用成员分析的真实 captured_at，且必须与发送快照对应 track.observed_at 相差不超过5毫秒。无法对应时不通知。

同类通知冷却不是交互开始/结束的状态机：页面和消息只表示在该观测时刻检测到正在操作，不表示整个实验周期持续操作。换仪器可单独触发。未知身份暂不发送实名消息，也不按显示名称猜测对应人。

## 权限

|角色|查看|处理告警|修改群路由|
|---|---|---|---|
|admin 管理员|全部管理事件、审计|全部|允许|
|manager 负责人|授权 labs、可选 instruments 范围|范围内|禁止|
|experimenter 实验人员|授权 person_ids 中自己的事件|禁止|禁止|
|viewer 只读查看|授权实验室/仪器事件|禁止|禁止|
|producer 检测结果提交者|禁止读取管理事件|仅允许提交授权摄像头证据|禁止|

每个实际人员配置独立 principal 和 token_env。实验人员示例 yxq 的 person_ids 默认空，必须填写人脸注册库的真实 person_id，不根据姓名猜测。负责人未设置 instruments 时可查看其授权实验室所有仪器；设置后限制到指定仪器。入口/口罩等无 instrument_id 的事件按实验室授权。

**权限范围是新增的通知管理模块**。原有监控、注册、录像、标注及 GPU 后端接口尚未统一接入这些令牌；不能把本轮实现描述为整个系统已经完成权限隔离。若要外部人员登录整个监控系统，需进一步统一网关认证和后端访问控制。

API均要求 `Authorization: Bearer <各角色自己的令牌>`：

- GET `/api/management/events`：最多100条授权事件、确认状态、群投递状态。
- POST `/api/management/ack`：`{"event_id":"..."}`，负责人或管理员确认已处理。
- GET/POST `/api/management/routes`：管理员读取/替换 `{"routes":[...]}`，持久保存。
- GET `/api/management/audit`：管理员审计，包含事件创建、处理、路由更改、成员证据提交。
- POST `/api/management/evidence`：仅 producer，口罩或交互结果提交。

## 飞书配置（目前等待机器人创建）

预留三个群路由：administrators、lab_managers、experimenters。实验人员群默认只接收未戴口罩提醒；群机器人通知由路由控制，群成员管理属于飞书。应用 API 的个人查看权限与群成员权限不是同一机制；不要在混合成员群里配置管理专用消息。

服务器环境变量：

|群|Webhook|签名密钥|
|---|---|---|
|管理员|LAB_FEISHU_ADMIN_WEBHOOK|LAB_FEISHU_ADMIN_SECRET|
|负责人|LAB_FEISHU_MANAGER_WEBHOOK|LAB_FEISHU_MANAGER_SECRET|
|实验人员|LAB_FEISHU_EXPERIMENTER_WEBHOOK|LAB_FEISHU_EXPERIMENTER_SECRET|

密钥、Webhook、角色令牌不写入代码、API响应或日志。管理路由只能保存这些环境变量的名称。飞书 endpoint 限定 HTTPS `open.feishu.cn/open-apis/bot/v2/hook/...`，按官方时间戳+签名规则调用。每秒最多发一条，超时5秒；HTTP成功但飞书业务返回非零也作为失败。

飞书Webhook没有本模块可用的端到端幂等确认：请求成功后连接断开、或投递成功而数据库提交前进程崩溃，重试可能重复送达。消息带 event_id 便于识别；本地事件和队列具备唯一键，不能声称飞书外部投递恰好一次。

官方协议：https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot

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

mask_state 可使用 worn/not_worn/unknown；只有 not_worn 且有效才产生事件。status 非 valid 代表无可用判断，不应伪造 not_worn。expires_at 最多有效到 published_at+5秒。主项目必须正在运行实时快照发布器，其 SnapshotTrackIndex 内仍保留原始快照；无索引、不匹配、时间变化、身份未知均不接收为实名告警。提交成功但 `accepted=false` 表示无新可通知事件，也可能是冷却/重复，并不表示检测人物戴了口罩。

仪器侧自动轮询已存在的 `lab-interaction-results-1.2` analyses；若采用主动提交，evidence_type=interaction，并提供 instrument_id、可选 instrument_name、is_interacting 和同样的时间/快照字段。instrument_id 必须用队友现有仪器配置的真实编号；不会硬编码“超声波净化器”的ID。

## 现场验收与回退

本轮验证：通知模块15项测试通过（含真实本地 HTTP 页面/401认证检查）；交接快照关联5项、联合演示11项回归通过。原有前端分离测试42项中41项通过，1项既有阈值断言失败：测试要求0.95，而 Git HEAD 地图配置已是0.75，本轮未改该配置。为运行已声明的Schema回归检查，补装了本地 `.venv` 缺失的 `jsonschema>=4.18,<5`，未更改requirements约束。`py_compile`与`git diff --check`通过。

1. 单人入口进入室内：只通知一次；仅路过入口、离开、回头需验证不误报。
2. 两人交叉：每条消息的人员身份和原始快照框人工核对；未知人员不套用 yxq。
3. 仪器：队友 true结果触发，地图候选/false/pending不触发；过期/错快照/改时间拒绝。
4. 口罩：队友真实结果 not_worn≥0.9触发；unknown、遮挡、低分不触发。
5. 权限：无Token=401，越权=403；实验人员只能看到已配置 person_id；负责人只能处理授权事件。
6. 飞书：机器人配置后，在授权管理群实测送达和状态；断网故障隔离、恢复重试、过期不补发。当前所有发送测试使用假发送器，未发送真实飞书消息。
7. 回退：去掉 `-EnableManagement`，设置 LAB_MANAGEMENT_ENABLED=0 后重启前端；不删除事件数据库、不改跟踪模型、不改队友交接Schema。
