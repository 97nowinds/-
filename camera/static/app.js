const videoGrid = document.querySelector("#videoGrid");
const peopleList = document.querySelector("#peopleList");
const serviceState = document.querySelector("#serviceState");
const cameraSummary = document.querySelector("#cameraSummary");
const identitySummary = document.querySelector("#identitySummary");
const handoffSummary = document.querySelector("#handoffSummary");
const floorMapSvg = document.querySelector("#floorMapSvg");
const floorMapStatic = document.querySelector("#floorMapStatic");
const floorMapTrailsLayer = document.querySelector("#floorMapTrails");
const floorMapPeopleLayer = document.querySelector("#floorMapPeople");
const floorMapMode = document.querySelector("#floorMapMode");
const floorMapCount = document.querySelector("#floorMapCount");
const floorMapPeopleList = document.querySelector("#floorMapPeopleList");
const API_BASE = (
  window.__LAB_API_BASE__ ||
  new URLSearchParams(location.search).get("api") ||
  location.origin
).replace(/\/$/, "");
const faceCaptureLink = document.querySelector("#faceCaptureLink");
if (faceCaptureLink) faceCaptureLink.href = `/faces?api=${encodeURIComponent(API_BASE)}`;
const remoteStream = location.hostname.startsWith("100.") ||
  new URLSearchParams(location.search).get("remote") === "1";
const SVG_NS = "http://www.w3.org/2000/svg";
const FLOOR_VIEWBOX = {width: 1200, height: 700};
const floorTrails = new Map();
let floorMapSignature = "";

async function fetchState() {
  const response = await fetch(`${API_BASE}/api/state`, {cache: "no-store"});
  if (!response.ok) throw new Error("状态接口不可用");
  return response.json();
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
    if (zone.kind !== "overlap") continue;
    const rect = mapRect(floorMap, zone);
    floorMapStatic.appendChild(createSvg("rect", {...rect, class: "map-overlap"}));
  }

  for (const fixture of floorMap.fixtures || []) {
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
      fixture.name,
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

  appendMapLabel(floorMapStatic, 600, 550, "主通道", "map-area-label");
}

function personLabel(person) {
  if (!person.identified) {
    const match = String(person.track_id || "").match(/^person_(\d+)$/);
    return match ? `目标 ${match[1]}（未注册）` : "未注册";
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
    const moved = !last || Math.hypot(person.x - last.x, person.y - last.y) >= 0.08;
    if (moved) history.push({x: person.x, y: person.y, at: now});
    floorTrails.set(person.track_id, history.filter((point) => now - point.at < 30000).slice(-30));
  }
  for (const [trackId, history] of floorTrails) {
    if (!activeIds.has(trackId) && (!history.length || now - history.at(-1).at > 10000)) {
      floorTrails.delete(trackId);
    }
  }
}

function renderFloorMap(floorMap) {
  if (!floorMap) return;
  renderFloorPlan(floorMap);
  updateFloorTrails(floorMap);
  floorMapMode.textContent = floorMap.calibrated ? "四点标定" : "示意映射";
  floorMapMode.className = floorMap.calibrated ? "calibrated" : "schematic";
  floorMapCount.textContent = `${floorMap.people.length} 人`;
  floorMapPeopleLayer.replaceChildren();
  floorMapTrailsLayer.replaceChildren();

  for (const [trackId, history] of floorTrails) {
    if (history.length < 2) continue;
    const points = history.map((point) => {
      const mapped = mapPoint(floorMap, [point.x, point.y]);
      return `${mapped.x},${mapped.y}`;
    }).join(" ");
    const active = floorMap.people.find((person) => person.track_id === trackId);
    floorMapTrailsLayer.appendChild(createSvg("polyline", {
      points,
      class: `map-trail ${active?.identified ? "identified" : "unknown"}`,
    }));
  }

  for (const person of floorMap.people) {
    const point = mapPoint(floorMap, [person.x, person.y]);
    const group = createSvg("g", {
      class: `map-person ${person.identified ? "identified" : "unknown"}`,
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
        <span>${escapeHtml(person.zone)} · ${escapeHtml(person.cameras.join(" + "))}</span>
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
      ? `${camera.role === "entrance_identity" ? "门口识别" : "YOLO 追踪"} · ${camera.tracked_people || 0} 人`
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
