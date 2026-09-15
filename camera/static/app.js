const videoGrid = document.querySelector("#videoGrid");
const peopleList = document.querySelector("#peopleList");
const serviceState = document.querySelector("#serviceState");
const cameraSummary = document.querySelector("#cameraSummary");
const identitySummary = document.querySelector("#identitySummary");
const handoffSummary = document.querySelector("#handoffSummary");
const environmentDevice = document.querySelector("#environmentDevice");
const environmentStatus = document.querySelector("#environmentStatus");
const temperatureValue = document.querySelector("#temperatureValue");
const temperatureState = document.querySelector("#temperatureState");
const humidityValue = document.querySelector("#humidityValue");
const humidityState = document.querySelector("#humidityState");
const flameCard = document.querySelector("#flameCard");
const flameValue = document.querySelector("#flameValue");
const flameState = document.querySelector("#flameState");
const floorMapSvg = document.querySelector("#floorMapSvg");
const floorMapStatic = document.querySelector("#floorMapStatic");
const floorMapTrailsLayer = document.querySelector("#floorMapTrails");
const floorMapPeopleLayer = document.querySelector("#floorMapPeople");
const floorMapMode = document.querySelector("#floorMapMode");
const floorMapCount = document.querySelector("#floorMapCount");
const floorMapPeopleList = document.querySelector("#floorMapPeopleList");
const dashboardTime = document.querySelector("#dashboardTime");
const dashboardDate = document.querySelector("#dashboardDate");
const eventFeed = document.querySelector("#eventFeed");
const recordingControlButton = document.querySelector("#recordingControlButton");
const recordingControlLabel = document.querySelector("#recordingControlLabel");
const recordingDialog = document.querySelector("#recordingDialog");
const recordingForm = document.querySelector("#recordingForm");
const recordingCloseButton = document.querySelector("#recordingCloseButton");
const recordingCancelButton = document.querySelector("#recordingCancelButton");
const recordingStartButton = document.querySelector("#recordingStartButton");
const recordingStopButton = document.querySelector("#recordingStopButton");
const recordingSubjectId = document.querySelector("#recordingSubjectId");
const recordingNotes = document.querySelector("#recordingNotes");
const recordingBanner = document.querySelector("#recordingBanner");
const recordingStatusTitle = document.querySelector("#recordingStatusTitle");
const recordingStatusDetail = document.querySelector("#recordingStatusDetail");
const recordingElapsed = document.querySelector("#recordingElapsed");
const recordingCameraList = document.querySelector("#recordingCameraList");
const recordingSessionDetails = document.querySelector("#recordingSessionDetails");
const recordingSessionId = document.querySelector("#recordingSessionId");
const recordingDirectory = document.querySelector("#recordingDirectory");
const recordingFrameCount = document.querySelector("#recordingFrameCount");
const recordingError = document.querySelector("#recordingError");
const API_BASE = (
  window.__LAB_API_BASE__ ||
  new URLSearchParams(location.search).get("api") ||
  location.origin
).replace(/\/$/, "");
const HARDWARE_URL = window.__LAB_HARDWARE_URL__ || null;
const faceCaptureLink = document.querySelector("#faceCaptureLink");
if (faceCaptureLink) faceCaptureLink.href = `/faces?api=${encodeURIComponent(API_BASE)}`;
const calibrationLink = document.querySelector("#calibrationLink");
if (calibrationLink) calibrationLink.href = `/annotate?api=${encodeURIComponent(API_BASE)}`;
const recordingReviewLink = document.querySelector("#recordingReviewLink");
if (recordingReviewLink) recordingReviewLink.href = `/recordings?api=${encodeURIComponent(API_BASE)}`;
const remoteStream = location.hostname.startsWith("100.") ||
  new URLSearchParams(location.search).get("remote") === "1";
const SVG_NS = "http://www.w3.org/2000/svg";
const FLOOR_VIEWBOX = {width: 1200, height: 700};
const floorTrails = new Map();
const floorTrailLastSeen = new Map();
let floorMapSignature = "";
let latestRecording = null;
let latestCameras = [];
let recordingMutationPending = false;

function updateDashboardClock() {
  const now = new Date();
  if (dashboardTime) {
    dashboardTime.textContent = now.toLocaleTimeString("zh-CN", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
  if (dashboardDate) {
    dashboardDate.textContent = now.toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      weekday: "short",
    });
  }
}

updateDashboardClock();
window.setInterval(updateDashboardClock, 1000);

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total % 3600 / 60);
  const remainder = total % 60;
  const base = `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  return hours ? `${String(hours).padStart(2, "0")}:${base}` : base;
}

function recordingFrames(recording) {
  return Object.values(recording?.streams || {}).reduce(
    (total, stream) => total + Number(stream?.frames || 0),
    0,
  );
}

function showRecordingError(message = "") {
  recordingError.textContent = message;
  recordingError.hidden = !message;
}

function renderRecording(recording, cameras) {
  latestRecording = recording || {active: false};
  latestCameras = cameras || [];
  const active = Boolean(latestRecording.active);
  const allReady = latestCameras.length > 0 && latestCameras.every((camera) => camera.status === "running");

  recordingControlButton.classList.toggle("active", active);
  recordingControlLabel.textContent = active
    ? `录像中 ${formatElapsed(latestRecording.elapsed_seconds)}`
    : "录像";
  recordingBanner.className = `recording-banner ${active ? "active" : allReady ? "ready" : "blocked"}`;
  recordingStatusTitle.textContent = active
    ? "正在同步录制"
    : allReady
      ? "摄像头已就绪"
      : "录像条件未满足";
  recordingStatusDetail.textContent = active
    ? `${latestRecording.subject_id || "未命名"} · 三路原始画面与时间戳`
    : allReady
      ? "填写录像编号后即可开始"
      : "所有已配置摄像头必须处于 running 状态";
  recordingElapsed.textContent = formatElapsed(latestRecording.elapsed_seconds);

  recordingCameraList.innerHTML = latestCameras.length
    ? latestCameras.map((camera) => `
      <div class="recording-camera-row">
        <span class="recording-camera-indicator ${escapeHtml(camera.status)}"></span>
        <strong>${escapeHtml(camera.id)}</strong>
        <span>${escapeHtml(camera.status)}</span>
        <span>${escapeHtml(camera.capture_fps || 0)} FPS</span>
      </div>
    `).join("")
    : `<span class="empty">尚未取得摄像头状态</span>`;

  recordingSessionDetails.hidden = !latestRecording.session_id;
  recordingSessionId.textContent = latestRecording.session_id || "—";
  recordingDirectory.textContent = latestRecording.directory || "—";
  recordingFrameCount.textContent = `${recordingFrames(latestRecording)} 帧`;
  recordingSubjectId.disabled = active || recordingMutationPending;
  recordingNotes.disabled = active || recordingMutationPending;
  recordingStartButton.hidden = active;
  recordingStartButton.disabled = !allReady || recordingMutationPending;
  recordingStopButton.hidden = !active;
  recordingStopButton.disabled = recordingMutationPending;
  recordingCancelButton.textContent = active ? "后台录制" : "关闭";
}

async function recordingRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `录像接口返回 ${response.status}`);
  return payload;
}

function openRecordingDialog() {
  showRecordingError();
  renderRecording(latestRecording, latestCameras);
  if (!recordingDialog.open) recordingDialog.showModal();
  if (!latestRecording?.active) recordingSubjectId.focus();
}

function closeRecordingDialog() {
  if (recordingDialog.open) recordingDialog.close();
}

recordingControlButton.addEventListener("click", openRecordingDialog);
recordingCloseButton.addEventListener("click", closeRecordingDialog);
recordingCancelButton.addEventListener("click", closeRecordingDialog);
recordingDialog.addEventListener("click", (event) => {
  if (event.target === recordingDialog) closeRecordingDialog();
});

recordingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!recordingForm.reportValidity() || recordingMutationPending) return;
  recordingMutationPending = true;
  showRecordingError();
  renderRecording(latestRecording, latestCameras);
  try {
    latestRecording = await recordingRequest("/api/recording/start", {
      method: "POST",
      body: JSON.stringify({
        subject_id: recordingSubjectId.value.trim(),
        notes: recordingNotes.value.trim(),
      }),
    });
  } catch (error) {
    showRecordingError(error.message);
  } finally {
    recordingMutationPending = false;
    renderRecording(latestRecording, latestCameras);
  }
});

recordingStopButton.addEventListener("click", async () => {
  if (recordingMutationPending || !latestRecording?.active) return;
  recordingMutationPending = true;
  showRecordingError();
  renderRecording(latestRecording, latestCameras);
  try {
    latestRecording = await recordingRequest("/api/recording/stop", {
      method: "POST",
      body: "{}",
    });
  } catch (error) {
    showRecordingError(error.message);
  } finally {
    recordingMutationPending = false;
    renderRecording(latestRecording, latestCameras);
  }
});

function floorMapLabel(item) {
  if (item?.id === "secondary_aisle") return "副通道";
  if (item?.id === "rear_service") return "侧向通道";
  return item?.name || "";
}

async function fetchState() {
  const [response, hardwareResponse] = await Promise.all([
    fetch(`${API_BASE}/api/state`, {cache: "no-store"}),
    HARDWARE_URL
      ? fetch(HARDWARE_URL, {cache: "no-store"}).catch(() => null)
      : Promise.resolve(null),
  ]);
  if (!response.ok) throw new Error("状态接口不可用");
  const state = await response.json();
  if (hardwareResponse?.ok) state.environment = await hardwareResponse.json();
  return state;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function identityText(identity) {
  if (!identity) return "搜索身份";
  const number = identity.person_number ? `${identity.person_number}号 ` : "";
  return `${number}${identity.name}`;
}

function createSvg(tag, attributes = {}, text = "") {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  if (text) element.textContent = text;
  return element;
}

function mapPoint(floorMap, point) {
  return {
    x: point[0] / floorMap.width_m * FLOOR_VIEWBOX.width,
    y: point[1] / floorMap.height_m * FLOOR_VIEWBOX.height,
  };
}

function mapRect(floorMap, item) {
  const origin = mapPoint(floorMap, [item.x, item.y]);
  return {
    x: origin.x,
    y: origin.y,
    width: item.width / floorMap.width_m * FLOOR_VIEWBOX.width,
    height: item.height / floorMap.height_m * FLOOR_VIEWBOX.height,
  };
}

function appendMapLabel(parent, x, y, text, className) {
  parent.appendChild(createSvg("text", {x, y, class: className, "text-anchor": "middle"}, text));
}

function renderFloorPlan(floorMap) {
  const signature = JSON.stringify({
    width_m: floorMap.width_m,
    height_m: floorMap.height_m,
    zones: floorMap.zones,
    fixtures: floorMap.fixtures,
    cameras: floorMap.cameras,
  });
  if (signature === floorMapSignature) return;
  floorMapSignature = signature;
  floorMapStatic.replaceChildren();

  floorMapStatic.appendChild(createSvg("rect", {
    x: 12, y: 12, width: 1176, height: 676, rx: 4, class: "map-room",
  }));

  for (const zone of floorMap.zones || []) {
    const rect = mapRect(floorMap, zone);
    if (zone.kind === "overlap") {
      floorMapStatic.appendChild(createSvg("rect", {...rect, class: "map-overlap"}));
    } else if (zone.id === "rear_service") {
      floorMapStatic.appendChild(createSvg("rect", {...rect, rx: 5, class: "map-side-aisle"}));
      appendMapLabel(
        floorMapStatic,
        rect.x + rect.width / 2,
        rect.y + rect.height / 2 + 10,
        "侧向通道",
        "map-side-aisle-label",
      );
    }
  }

  for (const fixture of floorMap.fixtures || []) {
    if (fixture.id === "rear_console") continue;
    const rect = mapRect(floorMap, fixture);
    floorMapStatic.appendChild(createSvg("rect", {
      ...rect,
      rx: 5,
      class: `map-fixture ${fixture.type || "fixture"}`,
    }));
    appendMapLabel(
      floorMapStatic,
      rect.x + rect.width / 2,
      rect.y + rect.height / 2 + 11,
      floorMapLabel(fixture),
      "map-fixture-label",
    );
  }

  for (const [cameraId, camera] of Object.entries(floorMap.cameras || {})) {
    const fov = (camera.fov || []).map((point) => {
      const mapped = mapPoint(floorMap, point);
      return `${mapped.x},${mapped.y}`;
    }).join(" ");
    if (fov) floorMapStatic.appendChild(createSvg("polygon", {points: fov, class: `map-fov ${cameraId}`}));
    const position = mapPoint(floorMap, camera.position);
    floorMapStatic.appendChild(createSvg("circle", {cx: position.x, cy: position.y, r: 24, class: `map-camera ${cameraId}`}));
    appendMapLabel(floorMapStatic, position.x, position.y + 10, camera.label, "map-camera-label");
  }

  const secondaryAisle = (floorMap.zones || []).find((zone) => zone.id === "secondary_aisle");
  if (secondaryAisle) {
    const rect = mapRect(floorMap, secondaryAisle);
    appendMapLabel(
      floorMapStatic,
      rect.x + rect.width / 2,
      rect.y + rect.height / 2 + 10,
      "副通道",
      "map-area-label",
    );
  }
  appendMapLabel(floorMapStatic, 600, 550, "主通道", "map-area-label");
}

function personLabel(person) {
  if (person.count_status === "duplicate_suppressed") return "疑似重复（不计数）";
  if (person.counted === false) return "待确认（不计数）";
  if (!person.identified) {
    return "未注册目标";
  }
  const number = person.person_number ? `${person.person_number}号 ` : "";
  return `${number}${person.name || "已确认"}`;
}

function updateFloorTrails(floorMap) {
  const now = Date.now();
  const activeIds = new Set();
  for (const person of floorMap.people || []) {
    activeIds.add(person.track_id);
    const history = floorTrails.get(person.track_id) || [];
    const last = history.at(-1);
    const lastSeenAt = floorTrailLastSeen.get(person.track_id);
    const breakBefore = Boolean(
      last && lastSeenAt && now - lastSeenAt > 4500
    );
    const moved = !last || Math.hypot(person.x - last.x, person.y - last.y) >= 0.08;
    if (moved || breakBefore) {
      history.push({
        x: person.x,
        y: person.y,
        at: now,
        source: person.position_source_camera,
        breakBefore,
      });
    }
    floorTrailLastSeen.set(person.track_id, now);
    floorTrails.set(person.track_id, history.filter((point) => now - point.at < 30000).slice(-30));
  }
  for (const [trackId, history] of floorTrails) {
    if (!activeIds.has(trackId) && (!history.length || now - history.at(-1).at > 10000)) {
      floorTrails.delete(trackId);
      floorTrailLastSeen.delete(trackId);
    }
  }
}

function renderFloorMap(floorMap) {
  if (!floorMap) return;
  renderFloorPlan(floorMap);
  updateFloorTrails(floorMap);
  floorMapMode.textContent = floorMap.calibrated ? "四点标定" : "示意映射";
  floorMapMode.className = floorMap.calibrated ? "calibrated" : "schematic";
  const countedPeople = Number.isFinite(Number(floorMap.counted_people))
    ? Number(floorMap.counted_people)
    : floorMap.people.filter((person) => person.counted !== false).length;
  floorMapCount.textContent = `${countedPeople} 人`;
  floorMapPeopleLayer.replaceChildren();
  floorMapTrailsLayer.replaceChildren();

  for (const [trackId, history] of floorTrails) {
    if (history.length < 2) continue;
    const active = floorMap.people.find((person) => person.track_id === trackId);
    const segments = [];
    let segment = [];
    for (const point of history) {
      if (point.breakBefore && segment.length) {
        segments.push(segment);
        segment = [];
      }
      segment.push(point);
    }
    if (segment.length) segments.push(segment);
    for (const pointsInSegment of segments.filter((points) => points.length >= 2)) {
      const points = pointsInSegment.map((point) => {
        const mapped = mapPoint(floorMap, [point.x, point.y]);
        return `${mapped.x},${mapped.y}`;
      }).join(" ");
      floorMapTrailsLayer.appendChild(createSvg("polyline", {
        points,
        class: `map-trail ${active?.identified ? "identified" : "unknown"}`,
      }));
    }
  }

  for (const person of floorMap.people) {
    const point = mapPoint(floorMap, [person.x, person.y]);
    const group = createSvg("g", {
      class: `map-person ${person.identified ? "identified" : "unknown"} ${person.position_estimated ? "held" : ""}`,
      transform: `translate(${point.x} ${point.y})`,
    });
    group.appendChild(createSvg("circle", {r: 42, class: "map-person-halo"}));
    group.appendChild(createSvg("circle", {r: 25, class: "map-person-dot"}));
    group.appendChild(createSvg("text", {x: 0, y: -38, class: "map-person-label", "text-anchor": "middle"}, personLabel(person)));
    floorMapPeopleLayer.appendChild(group);
  }

  floorMapPeopleList.innerHTML = floorMap.people.length
    ? floorMap.people.map((person) => `
      <div class="floor-person-row">
        <span class="floor-person-indicator ${person.identified ? "identified" : "unknown"}"></span>
        <strong>${escapeHtml(personLabel(person))}</strong>
        <span class="floor-person-location">${escapeHtml(["内侧通道", "后端操作区"].includes(person.zone) ? "侧向通道" : person.zone)} · ${escapeHtml(person.cameras.join(" + "))}${person.position_estimated ? ` · 定位保持 ${Number(person.position_age_seconds || 0).toFixed(1)}s` : ""}</span>
        <span class="floor-person-lock ${person.identity_lock_status === "locked" ? "locked" : "visual"}">${person.identity_lock_status === "locked" ? "已锁定" : person.count_status === "duplicate_suppressed" ? "跨摄疑似重复 · 不计数" : person.counted === false ? "临时候选 · 不计数" : "视觉追踪"}</span>
      </div>
    `).join("")
    : `<span class="empty">当前没有活动目标</span>`;
}

function modeText(camera) {
  if (camera.tracker_error) return "YOLO 追踪器异常";
  if (!camera.identity) {
    if (camera.role === "entrance_identity" && !camera.face_recognition_ready) {
      return "ArcFace 模型未就绪";
    }
    return camera.tracking
      ? `${camera.role === "entrance_identity" ? "门口识别" : "YOLO 追踪"} · ${camera.counted_people ?? camera.tracked_people ?? 0} 人`
      : "YOLO 等待人员进入";
  }
  return ["handoff", "overlap_handoff", "transition_handoff"].includes(camera.identity.identity_source)
    ? `跨镜头继承自 ${camera.identity.handoff_from_camera}`
    : "本摄像头人脸确认";
}

function renderPeopleList(people) {
  peopleList.innerHTML = people.length
    ? people.map((person) => `
      <div class="person-row">
        <strong>${escapeHtml(person.label)}号 ${escapeHtml(person.name)}</strong>
        <span>${escapeHtml(person.id)} · ${escapeHtml(person.sample_count)}张样本</span>
      </div>
    `).join("")
    : `<div class="empty">暂无注册人员</div>`;
}

function renderEnvironment(environment = {}) {
  const statusLabels = {
    disabled: "未配置",
    disconnected: "等待连接",
    connected: "串口已连接",
    online: "在线",
    alarm: "火情告警",
    stale: "数据超时",
    error: "连接异常",
  };
  const temperature = Number(environment.temperature);
  const humidity = Number(environment.humidity);
  const hasTemperature = environment.temperature != null && Number.isFinite(temperature);
  const hasHumidity = environment.humidity != null && Number.isFinite(humidity);
  environmentDevice.textContent = environment.port
    ? `${environment.device_id || "HAFS-STM32"} · ${environment.port} · ${environment.protocol || "等待报文"}`
    : environment.device_id || "HAFS 硬件未配置";
  environmentStatus.textContent = statusLabels[environment.status] || environment.status || "未知";
  environmentStatus.className = `environment-status ${environment.status || "disabled"}`;
  temperatureValue.textContent = hasTemperature ? `${temperature.toFixed(1)} ℃` : "--.- ℃";
  humidityValue.textContent = hasHumidity ? `${humidity.toFixed(1)} %` : "--.- %";
  const dhtText = environment.dht_valid ? "DHT 校验正常" : "DHT 数据无效或未收到";
  temperatureState.textContent = dhtText;
  humidityState.textContent = dhtText;
  flameCard.classList.toggle("alarm", Boolean(environment.flame_alarm));
  flameValue.textContent = environment.flame_alarm ? "危险" : environment.last_seen_at ? "安全" : "未知";
  flameState.textContent = environment.flame_alarm
    ? "现场蜂鸣器与 LED 已触发"
    : environment.age_seconds == null
      ? "等待硬件状态"
      : `${environment.age_seconds.toFixed(1)} 秒前更新`;
}

function renderEventFeed(state) {
  if (!eventFeed) return;
  const now = new Date().toLocaleTimeString("zh-CN", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const rows = [];
  const environment = state.environment || {};
  const activePeople = state.floor_map?.people || [];

  if (environment.flame_alarm) {
    rows.push({time: now, text: "检测到火焰触点告警，请立即核查", tag: "环境告警", tone: "alarm"});
  }

  for (const camera of state.cameras || []) {
    if (camera.status !== "running") {
      rows.push({time: now, text: `${camera.name || camera.id} 当前状态：${camera.status}`, tag: "设备提示", tone: "warning"});
      continue;
    }
    if (camera.identity) {
      rows.push({time: now, text: `${camera.name || camera.id} 已识别 ${identityText(camera.identity)}`, tag: "人员识别", tone: "normal"});
    }
  }

  const handoffCamera = (state.cameras || []).find((camera) =>
    ["handoff", "overlap_handoff", "transition_handoff"].includes(camera.identity?.identity_source),
  );
  if (handoffCamera) {
    rows.push({
      time: now,
      text: `${handoffCamera.identity.handoff_from_camera} → ${handoffCamera.id} 身份继承完成`,
      tag: "跨镜追踪",
      tone: "normal",
    });
  }

  const rawInteractions = state.interaction_events || state.instrument_interaction?.events || state.interactions || [];
  for (const interaction of Array.isArray(rawInteractions) ? rawInteractions.slice(0, 3) : []) {
    rows.push({
      time: String(interaction.time || interaction.timestamp || now).slice(-8),
      text: interaction.message || interaction.description || interaction.event || "检测到仪器交互",
      tag: "仪器交互",
      tone: interaction.level === "alarm" ? "alarm" : "normal",
    });
  }

  if (!rows.length) {
    rows.push(
      {time: now, text: `${state.cameras?.length || 0} 路监控状态持续检测中`, tag: "系统运行", tone: "normal"},
      {time: now, text: activePeople.length ? `实验室内当前检测到 ${activePeople.length} 人` : "当前区域未检测到活动目标", tag: "区域状态", tone: "normal"},
      {time: now, text: "仪器交互信息区域已预留", tag: "仪器交互", tone: "normal"},
    );
  }

  eventFeed.innerHTML = rows.slice(0, 7).map((row) => `
    <div class="event-row ${escapeHtml(row.tone)}">
      <time>${escapeHtml(row.time)}</time>
      <strong>${escapeHtml(row.text)}</strong>
      <span>${escapeHtml(row.tag)}</span>
    </div>
  `).join("");
}

function renderInitial(state) {
  videoGrid.innerHTML = state.cameras.map((camera) => `
    <article class="camera-card" data-camera-id="${escapeHtml(camera.id)}">
      <header>
        <div>
          <h2>${escapeHtml(camera.name)}</h2>
          <span>${escapeHtml(camera.source)}</span>
        </div>
        <span class="camera-badge ${escapeHtml(camera.status)}">${escapeHtml(camera.status)}</span>
      </header>
      <div class="camera-viewport">
        <img src="${API_BASE}/video/${encodeURIComponent(camera.id)}${remoteStream ? "?remote=1" : ""}" alt="${escapeHtml(camera.name)}">
      </div>
      <footer>
        <strong class="camera-identity">${escapeHtml(identityText(camera.identity))}</strong>
        <span class="camera-mode">${escapeHtml(modeText(camera))}</span>
        <span class="camera-fps">${escapeHtml(camera.fps)} FPS</span>
      </footer>
    </article>
  `).join("");

  renderPeopleList(state.people || []);
}

function update(state) {
  if (videoGrid.children.length !== state.cameras.length) renderInitial(state);
  renderPeopleList(state.people || []);
  const runningCount = state.cameras.filter((camera) => camera.status === "running").length;
  cameraSummary.textContent = `${runningCount}/${state.cameras.length} 路正常`;
  const identified = state.cameras.filter((camera) => camera.identity);
  const uniquePeople = [...new Set(identified.map((camera) => identityText(camera.identity)))];
  identitySummary.textContent = uniquePeople.length ? uniquePeople.join("、") : "尚未确认";
  const handoff = identified.find((camera) => ["handoff", "overlap_handoff", "transition_handoff"].includes(camera.identity.identity_source));
  handoffSummary.textContent = handoff
    ? `${handoff.identity.handoff_from_camera} → ${handoff.id} · ${identityText(handoff.identity)}`
    : "等待重叠区继承";
  renderFloorMap(state.floor_map);
  renderRecording(state.recording, state.cameras);
  renderEnvironment(state.environment);
  renderEventFeed(state);

  for (const camera of state.cameras) {
    const card = document.querySelector(`[data-camera-id="${CSS.escape(camera.id)}"]`);
    if (!card) continue;
    const badge = card.querySelector(".camera-badge");
    badge.textContent = camera.status;
    badge.className = `camera-badge ${camera.status}`;
    card.querySelector(".camera-identity").textContent = identityText(camera.identity);
    card.querySelector(".camera-mode").textContent = modeText(camera);
    const latency = camera.processing_latency_ms == null ? "-" : camera.processing_latency_ms;
    card.querySelector(".camera-fps").textContent = `采 ${camera.capture_fps || 0}/处 ${camera.fps || 0} FPS · ${latency}ms · 丢 ${camera.dropped_frames || 0}`;
  }
  serviceState.textContent = "服务在线";
  serviceState.className = "service-state online";
}

async function refresh() {
  try {
    update(await fetchState());
  } catch (error) {
    serviceState.textContent = "服务断开";
    serviceState.className = "service-state error";
  } finally {
    // 与后端 200 ms 地图发布节奏对齐；视频流仍由独立 MJPEG 接口连续输出。
    window.setTimeout(refresh, 200);
  }
}

refresh();
