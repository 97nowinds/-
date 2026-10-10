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
const INTERACTION_URL = "/api/interaction-results/latest";
const resultContract = window.__LAB_RESULT_CONTRACT__ || "";
let latestAnalyses = [];
const interactionDeadlines = new Map();

function currentInteractionEvents() {
  if (!resultContract) return latestInteractionEvents;
  return latestInteractionEvents.filter((item) =>
    performance.now() < (interactionDeadlines.get(item.analysis_id) || 0));
}

function evidenceTime(value) {
  const parsed = new Date(value);
  return Number.isFinite(parsed.getTime())
    ? parsed.toLocaleTimeString("zh-CN", {hour12: false}) : "时间未知";
}
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
const floorTrailPendingBreak = new Set();
let lastRenderedReplayCycle = null;
let floorMapSignature = "";
let latestRecording = null;
let latestCameras = [];
let recordingMutationPending = false;
let peopleListSignature = "";
let floorPeopleListSignature = "";
let eventFeedSignature = "";
let latestInteractionEvents = [];
let latestInteractionErrors = [];
let instrumentAnchors = [];
let instrumentAnchorsFetchedAt = 0;
let interactionFetchPending = false;
let interactionLastFetchMs = 0;
let synchronizedReplayEnabled = false;
let synchronizedReplayBusy = false;
let synchronizedReplayTimer = null;
let synchronizedReplaySequence = 0;
let synchronizedReplayCameraIds = [];
const synchronizedReplayObjectUrls = new Map();

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

// The map backend publishes the newest confirmed track state every 50 ms.
// Video preview has its own stream. Map positions and the personnel panel
// refresh from the same status snapshot. A 2 Hz compact status refresh keeps
// controls responsive without making JSON parsing compete with MJPEG decode.
const DASHBOARD_REFRESH_INTERVAL_MS = 500;
const SERVICE_DISCONNECT_FAILURES = 3;
let consecutiveStateFailures = 0;

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
    fetch(`${API_BASE}/api/dashboard`, {cache: "no-store"}),
    HARDWARE_URL
      ? fetch(HARDWARE_URL, {cache: "no-store"}).catch(() => null)
      : Promise.resolve(null),
  ]);
  if (!response.ok) throw new Error("状态接口不可用");
  const state = await response.json();
  if (hardwareResponse?.ok) {
    const environment = await hardwareResponse.json().catch(() => null);
    if (environment) state.environment = environment;
  }
  state.interaction_events = currentInteractionEvents();
  state.interaction_errors = latestInteractionErrors;
  state.interaction_analyses = latestAnalyses;
  return state;
}

function refreshInteractionResults() {
  const now = Date.now();
  if (now - instrumentAnchorsFetchedAt >= 30000) {
    instrumentAnchorsFetchedAt = now;
    fetch("/api/instruments/anchors", {cache: "no-store"})
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("instrument anchors unavailable")))
      .then((payload) => { instrumentAnchors = Array.isArray(payload?.instruments) ? payload.instruments : []; })
      .catch(() => { instrumentAnchors = []; });
  }
  if (interactionFetchPending || now - interactionLastFetchMs < 1000) return;
  interactionFetchPending = true;
  interactionLastFetchMs = now;
  const resultUrl = resultContract === "lab-interaction-results-1.2"
    ? `${INTERACTION_URL}?contract=${encodeURIComponent(resultContract)}` : INTERACTION_URL;
  const requestedAt = performance.now();
  fetch(resultUrl, {cache: "no-store"})
    .then((response) => response.ok ? response.json() : Promise.reject(new Error("interaction unavailable")))
    .then((payload) => {
      if (resultContract && payload?.result_contract !== resultContract) throw new Error("result contract unavailable");
      latestInteractionEvents = Array.isArray(payload) ? payload : Array.isArray(payload?.results) ? payload.results : [];
      latestInteractionErrors = Array.isArray(payload?.errors) ? payload.errors : [];
      latestAnalyses = Array.isArray(payload?.analyses) ? payload.analyses : [];
      if (resultContract) {
        for (const item of latestInteractionEvents) {
          const remaining = Math.min(5000, Date.parse(item.expires_at) - Date.parse(payload.generated_at))
            - (performance.now() - requestedAt);
          const deadline = performance.now() + Math.max(0, Number.isFinite(remaining) ? remaining : 0);
          interactionDeadlines.set(item.analysis_id,
            Math.min(interactionDeadlines.get(item.analysis_id) ?? deadline, deadline));
        }
      }
    })
    .catch(() => {
      latestInteractionEvents = [];
      latestInteractionErrors = [];
      latestAnalyses = resultContract ? [{status: "unavailable", reason_code: "receiver_unavailable"}] : [];
    })
    .finally(() => {
      interactionFetchPending = false;
    });
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
  const trackLabel = person.global_track_id || person.track_id || "";
  if (person.count_status === "duplicate_suppressed") return `${trackLabel} 疑似重复（不计数）`;
  if (person.counted === false) return `${trackLabel} 待确认（不计数）`;
  if (!person.identified) {
    return `${trackLabel} 未注册目标`;
  }
  const number = person.person_number ? `${person.person_number}号 ` : "";
  return `${trackLabel} ${number}${person.name || "已确认"}`;
}

function updateFloorTrails(floorMap) {
  const now = Date.now();
  const activeIds = new Set();
  for (const person of floorMap.people || []) {
    activeIds.add(person.track_id);
    if (person.map_visible === false) {
      floorTrailPendingBreak.add(person.track_id);
      continue;
    }
    const history = floorTrails.get(person.track_id) || [];
    const last = history.at(-1);
    const lastSeenAt = floorTrailLastSeen.get(person.track_id);
    const breakBefore = Boolean(
      last && (
        (lastSeenAt && now - lastSeenAt > 4500) ||
        person.position_discontinuity ||
        floorTrailPendingBreak.has(person.track_id) ||
        (last.source && person.position_source_camera &&
          last.source !== person.position_source_camera &&
          Math.hypot(person.x - last.x, person.y - last.y) > 1.5)
      )
    );
    const moved = !last || breakBefore || Math.hypot(person.x - last.x, person.y - last.y) >= 0.08;
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
    floorTrailPendingBreak.delete(person.track_id);
    floorTrails.set(person.track_id, history.filter((point) => now - point.at < 30000).slice(-30));
  }
  for (const [trackId, history] of floorTrails) {
    if (!activeIds.has(trackId) && (!history.length || now - history.at(-1).at > 10000)) {
      floorTrails.delete(trackId);
      floorTrailLastSeen.delete(trackId);
      floorTrailPendingBreak.delete(trackId);
    }
  }
}

function renderFloorMap(floorMap) {
  if (!floorMap) return;
  renderFloorPlan(floorMap);
  updateFloorTrails(floorMap);
  floorMapMode.textContent = floorMap.calibrated ? "四点标定 · 实时更新" : "示意映射 · 实时更新";
  floorMapMode.className = floorMap.calibrated ? "calibrated" : "schematic";
  const countedPeople = Number.isFinite(Number(floorMap.counted_people))
    ? Number(floorMap.counted_people)
    : floorMap.people.filter((person) => person.counted !== false).length;
  floorMapCount.textContent = `${countedPeople} 人`;
  floorMapPeopleLayer.replaceChildren();
  floorMapTrailsLayer.replaceChildren();

  for (const person of floorMap.people) {
    if (person.map_visible === false) continue;
    const point = mapPoint(floorMap, [person.x, person.y]);
    const estimatedPositionClass = person.localization_state === "held"
      ? "held"
      : person.localization_state === "predicted"
        ? "estimated"
      : person.position_estimated
        ? "estimated"
        : "";
    const group = createSvg("g", {
      class: `map-person ${person.identified ? "identified" : "unknown"} ${estimatedPositionClass}`,
      transform: `translate(${point.x} ${point.y})`,
      "data-map-track": person.track_id,
    });
    group.appendChild(createSvg("circle", {r: 42, class: "map-person-halo"}));
    group.appendChild(createSvg("circle", {r: 25, class: "map-person-dot"}));
    group.appendChild(createSvg("text", {x: 0, y: -38, class: "map-person-label", "text-anchor": "middle"}, personLabel(person)));
    floorMapPeopleLayer.appendChild(group);
  }

  const nextFloorPeopleListSignature = JSON.stringify(floorMap.people.map((person) => [
    person.track_id,
    personLabel(person),
    person.zone,
    person.cameras,
    Boolean(person.position_estimated),
    person.identity_lock_status,
    person.count_status,
    person.counted,
    person.handoff_waiting,
    person.map_visible,
  ]));
  if (nextFloorPeopleListSignature === floorPeopleListSignature) return;
  floorPeopleListSignature = nextFloorPeopleListSignature;
  floorMapPeopleList.innerHTML = floorMap.people.length
    ? floorMap.people.map((person) => `
      <div class="floor-person-row">
        <span class="floor-person-indicator ${person.identified ? "identified" : "unknown"}"></span>
        <strong>${escapeHtml(personLabel(person))}</strong>
        <span class="floor-person-location">${escapeHtml(person.map_visible === false ? "位置待确认" : ["内侧通道", "后端操作区"].includes(person.zone) ? "侧向通道" : person.zone)} · ${escapeHtml(person.cameras.join(" + "))}${person.position_source === "instrument_anchor" ? ` · ${escapeHtml(person.instrument_name)}仪器定位 ${person.position_score.toFixed(2)}` : person.handoff_waiting ? " · 跨摄定位中" : person.localization_state === "predicted" ? " · 短暂预测" : person.localization_state === "held" ? " · 定位保持" : person.position_estimated ? " · 脚点估算" : ""}</span>
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
    if (!camera.tracking) return "YOLO 等待人员进入";
    const tracked = camera.tracked_people ?? 0;
    const counted = camera.counted_people ?? tracked;
    const label = camera.role === "entrance_identity" ? "门口识别" : "YOLO 追踪";
    return tracked === counted
      ? `${label} · ${tracked} 人`
      : `${label} · ${tracked} 轨 / ${counted} 已确认`;
  }
  return ["handoff", "overlap_handoff", "transition_handoff"].includes(camera.identity.identity_source)
    ? `跨镜头继承自 ${camera.identity.handoff_from_camera}`
    : "本摄像头人脸确认";
}

function renderPeopleList(people) {
  const nextSignature = JSON.stringify(people.map((person) => [
    person.id,
    person.label,
    person.name,
    person.sample_count,
  ]));
  if (nextSignature === peopleListSignature) return;
  peopleListSignature = nextSignature;
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
    const interactionSubject = interaction.handoff_match_status
      ? interaction.display_name || `匿名轨迹 ${interaction.global_id || interaction.track_id || "未知"}`
      : interaction.display_name || interaction.person_name || interaction.track_id || "目标";
    const interactionAction = interaction.interaction_label || interaction.message || interaction.description || interaction.event || "检测到交互";
    const interactionInstrument = interaction.instrument_name || interaction.instrument_id || "仪器";
    rows.push({
      time: evidenceTime(interaction.observed_at || interaction.captured_at || interaction.time || interaction.timestamp || now),
      text: `${interactionSubject} ${interactionAction} ${interactionInstrument}`,
      tag: interaction.camera_id ? `仪器交互 · ${interaction.camera_id}` : "仪器交互",
      tone: interaction.level === "alarm" ? "alarm" : "normal",
    });
  }

  for (const error of Array.isArray(state.interaction_errors) ? state.interaction_errors.slice(0, 3) : []) {
    rows.push({
      time: String(error.occurred_at || now).slice(11, 19),
      text: `仪器分析失败：${error.code || error.message || "unknown_error"}`,
      tag: error.camera_id ? `仪器交互 · ${error.camera_id}` : "仪器交互",
      tone: "warning",
    });
  }

  const statusLabels = {pending: "等待分析", unavailable: "暂不可分析", error: "分析失败", stale: "证据超龄", expired: "结果已过期"};
  for (const analysis of (state.interaction_analyses || []).slice(0, 3)) {
    if (analysis.status === "valid" && analysis.is_interacting !== false) continue;
    rows.push({time: evidenceTime(analysis.captured_at || now),
      text: `${analysis.display_name || analysis.track_id || "仪器交互"}：${analysis.status === "valid" ? "模型判断未交互" : statusLabels[analysis.status] || "状态未知"} · ${analysis.reason_code || ""}`,
      tag: "分析状态", tone: "warning"});
  }

  if (!rows.length) {
    rows.push(
      {time: now, text: `${state.cameras?.length || 0} 路监控状态持续检测中`, tag: "系统运行", tone: "normal"},
      {time: now, text: activePeople.length ? `实验室内当前检测到 ${activePeople.length} 人` : "当前区域未检测到活动目标", tag: "区域状态", tone: "normal"},
      {time: now, text: "仪器交互信息通道已就绪", tag: "仪器交互", tone: "normal"},
    );
  }

  const reservedChannels = [
    ["人员进入事件", "人员识别"],
    ["跨区域轨迹事件", "跨镜追踪"],
    ["仪器操作事件", "仪器交互"],
    ["温湿度异常事件", "环境监测"],
    ["火情与设备异常事件", "安全告警"],
    ["系统状态更新", "运行状态"],
  ];
  let reservedIndex = 0;
  while (rows.length < 9) {
    const channel = reservedChannels[reservedIndex % reservedChannels.length];
    rows.push({time: "--:--:--", text: `${channel[0]} · 等待数据`, tag: channel[1], tone: "placeholder"});
    reservedIndex += 1;
  }

  const nextSignature = JSON.stringify(rows.slice(0, 9).map((row) => [row.text, row.tag, row.tone]));
  if (nextSignature === eventFeedSignature) return;
  eventFeedSignature = nextSignature;

  eventFeed.innerHTML = rows.slice(0, 9).map((row) => `
    <div class="event-row ${escapeHtml(row.tone)}">
      <time>${escapeHtml(row.time)}</time>
      <strong>${escapeHtml(row.text)}</strong>
      <span>${escapeHtml(row.tag)}</span>
    </div>
  `).join("");
}

function renderInitial(state) {
  if (!videoGrid) return;
  const synchronizedReplay = state.cameras.length > 0 &&
    state.cameras.every((camera) => camera.replay?.enabled);
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
        <img data-camera-preview="${escapeHtml(camera.id)}" ${synchronizedReplay ? "" : `src="${API_BASE}/video/${encodeURIComponent(camera.id)}${remoteStream ? "?remote=1" : ""}"`} alt="${escapeHtml(camera.name)}">
      </div>
      <footer>
        <strong class="camera-identity">${escapeHtml(identityText(camera.identity))}</strong>
        <span class="camera-mode">${escapeHtml(modeText(camera))}</span>
        <span class="camera-fps">${escapeHtml(camera.preview_fps || camera.fps)} FPS</span>
      </footer>
    </article>
  `).join("");

  renderPeopleList(state.people || []);
}

async function refreshSynchronizedReplay() {
  if (!synchronizedReplayEnabled || synchronizedReplayBusy) return;
  synchronizedReplayBusy = true;
  const syncId = `preview_${Date.now()}_${++synchronizedReplaySequence}`;
  try {
    const frames = await Promise.all(synchronizedReplayCameraIds.map(async (cameraId) => {
      const response = await fetch(
        `${API_BASE}/api/replay/synchronized-frame/${encodeURIComponent(cameraId)}.jpg?sync_id=${encodeURIComponent(syncId)}`,
        {cache: "no-store"},
      );
      if (!response.ok) throw new Error(`sync preview ${cameraId}: HTTP ${response.status}`);
      const url = URL.createObjectURL(await response.blob());
      const preload = new Image();
      preload.src = url;
      await preload.decode();
      return {cameraId, url};
    }));

    // Decode the whole batch before changing any card. The three panels then
    // advance in one browser render instead of through independent MJPEG queues.
    for (const {cameraId, url} of frames) {
      const image = document.querySelector(`[data-camera-preview="${CSS.escape(cameraId)}"]`);
      if (image) image.src = url;
    }
    for (const {cameraId, url} of frames) {
      const previous = synchronizedReplayObjectUrls.get(cameraId);
      synchronizedReplayObjectUrls.set(cameraId, url);
      if (previous) URL.revokeObjectURL(previous);
    }
  } catch (error) {
    // A replay service restart is transient; the next synchronized batch retries.
  } finally {
    synchronizedReplayBusy = false;
    if (synchronizedReplayEnabled) {
      synchronizedReplayTimer = window.setTimeout(refreshSynchronizedReplay, 100);
    }
  }
}

function configureSynchronizedReplay(state) {
  if (!videoGrid) return;
  const enabled = state.cameras.length > 0 &&
    state.cameras.every((camera) => camera.replay?.enabled);
  synchronizedReplayCameraIds = state.cameras.map((camera) => camera.id);
  if (enabled === synchronizedReplayEnabled) return;
  synchronizedReplayEnabled = enabled;
  if (synchronizedReplayTimer) window.clearTimeout(synchronizedReplayTimer);
  synchronizedReplayTimer = null;
  if (enabled) {
    refreshSynchronizedReplay();
    return;
  }
  for (const url of synchronizedReplayObjectUrls.values()) URL.revokeObjectURL(url);
  synchronizedReplayObjectUrls.clear();
}

function update(state) {
  if (videoGrid && videoGrid.children.length !== state.cameras.length) renderInitial(state);
  configureSynchronizedReplay(state);
  const replayCycles = state.cameras.map((camera) => camera.replay?.enabled ? camera.replay.cycle : null);
  if (replayCycles.length && replayCycles.every((cycle) => cycle != null && cycle === replayCycles[0])) {
    const cycle = replayCycles[0];
    if (lastRenderedReplayCycle != null && cycle !== lastRenderedReplayCycle) {
      floorTrails.clear();
      floorTrailLastSeen.clear();
      floorTrailPendingBreak.clear();
    }
    lastRenderedReplayCycle = cycle;
  }
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
  renderFloorMap(InstrumentLocation.apply(state.floor_map, latestInteractionEvents, instrumentAnchors));
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
    card.querySelector(".camera-fps").textContent = `采 ${camera.capture_fps || 0}/显 ${camera.preview_fps || 0}/算 ${camera.fps || 0} FPS · ${latency}ms · 丢 ${camera.dropped_frames || 0}`;
  }
}

async function refresh() {
  try {
    refreshInteractionResults();
    let state;
    try {
      state = await fetchState();
    } catch (error) {
      consecutiveStateFailures += 1;
      const disconnected = consecutiveStateFailures >= SERVICE_DISCONNECT_FAILURES;
      serviceState.textContent = disconnected ? "服务断开" : "刷新延迟";
      serviceState.className = `service-state ${disconnected ? "error" : "delayed"}`;
      return;
    }
    consecutiveStateFailures = 0;
    update(state);
    serviceState.textContent = "服务在线";
    serviceState.className = "service-state online";
  } catch (error) {
    // A rendering error does not indicate that the state API is disconnected.
    console.error("监控页面刷新失败", error);
  } finally {
    // Video preview uses its own synchronized loop.  This is only state/map
    // refresh and must stay below the backend's 50 ms publication rate.
    window.setTimeout(refresh, DASHBOARD_REFRESH_INTERVAL_MS);
  }
}

refresh();
