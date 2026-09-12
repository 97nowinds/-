const API_BASE = (
  window.__LAB_API_BASE__ ||
  new URLSearchParams(location.search).get("api") ||
  location.origin
).replace(/\/$/, "");
const monitorLink = document.querySelector("#monitorLink");
const serviceState = document.querySelector("#reviewServiceState");
const sessionSelect = document.querySelector("#recordingSessionSelect");
const videoGrid = document.querySelector("#recordedVideoGrid");
const reviewError = document.querySelector("#reviewError");
const sourceSelect = document.querySelector("#handoffSourceCamera");
const targetSelect = document.querySelector("#handoffTargetCamera");
const personIdInput = document.querySelector("#handoffPersonId");
const notesInput = document.querySelector("#handoffNotes");
const sourceValue = document.querySelector("#sourceMarkerValue");
const targetValue = document.querySelector("#targetMarkerValue");
const gapPreview = document.querySelector("#handoffGapPreview");
const form = document.querySelector("#handoffForm");
const handoffList = document.querySelector("#handoffList");
const suggestions = document.querySelector("#handoffSuggestions");
let sessions = [];
let currentSession = null;
let sourceTime = null;
let targetTime = null;
const frameReview = new Map();

monitorLink.href = `/?api=${encodeURIComponent(API_BASE)}`;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function showError(message = "") {
  reviewError.textContent = message;
  reviewError.hidden = !message;
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `接口返回 ${response.status}`);
  return payload;
}

function sessionPath(suffix = "") {
  return `/api/recordings/${encodeURIComponent(currentSession.subject_id)}/${encodeURIComponent(currentSession.session_id)}${suffix}`;
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(3).padStart(6, "0")}`;
}

function describeGap(seconds) {
  const value = Number(seconds) || 0;
  return value < 0
    ? `重叠 ${Math.abs(value).toFixed(3)} 秒`
    : `盲区/通行 ${value.toFixed(3)} 秒`;
}

function videoFor(cameraId) {
  return document.querySelector(`video[data-camera-id="${CSS.escape(cameraId)}"]`);
}

function selectedMediaTime(cameraId) {
  const state = frameReview.get(cameraId);
  if (state?.active) return state.frameIndex / state.fps;
  return videoFor(cameraId)?.currentTime ?? null;
}

function showReviewFrame(card, requestedIndex) {
  const cameraId = card.dataset.cameraId;
  const state = frameReview.get(cameraId);
  if (!state) return;
  state.active = true;
  state.frameIndex = Math.max(0, Math.min(state.frames - 1, Math.round(requestedIndex)));
  const video = card.querySelector("video");
  const panel = card.querySelector(".recorded-frame-review");
  const slider = panel.querySelector("input");
  video.pause();
  video.hidden = true;
  panel.hidden = false;
  slider.value = state.frameIndex;
  panel.querySelector("img").src = `${API_BASE}${sessionPath(`/frame/${encodeURIComponent(cameraId)}`)}?index=${state.frameIndex}`;
  card.querySelector(".video-current-time").textContent = `${formatTime(state.frameIndex / state.fps)} · 帧 ${state.frameIndex}`;
}

function resetMarkers() {
  sourceTime = null;
  targetTime = null;
  sourceValue.textContent = "尚未标记";
  targetValue.textContent = "尚未标记";
  gapPreview.textContent = "等待两个时间点";
}

function updateGapPreview() {
  if (sourceTime == null || targetTime == null) {
    gapPreview.textContent = "等待两个时间点";
    return;
  }
  gapPreview.textContent = `视频相对时间差约 ${(targetTime - sourceTime).toFixed(3)} 秒；保存后按各路时间戳计算真实 gap`;
}

function renderVideos() {
  const streams = Object.values(currentSession?.streams || {}).filter((stream) => stream.video_file);
  frameReview.clear();
  videoGrid.innerHTML = streams.length ? streams.map((stream) => `
    <article class="recorded-video-card" data-camera-id="${escapeHtml(stream.camera_id)}">
      <header><strong>${escapeHtml(stream.camera_id)}</strong><span>${escapeHtml(stream.frames)} 帧 · ${escapeHtml(stream.fps)} FPS</span></header>
      <video data-camera-id="${escapeHtml(stream.camera_id)}" controls preload="metadata" crossorigin="anonymous"
        src="${API_BASE}${sessionPath(`/video/${encodeURIComponent(stream.camera_id)}`)}"></video>
      <div class="recorded-frame-review" hidden>
        <img alt="${escapeHtml(stream.camera_id)} 逐帧复核画面">
        <input type="range" min="0" max="${Math.max(0, Number(stream.frames) - 1)}" value="0" step="1" aria-label="${escapeHtml(stream.camera_id)} 帧位置">
      </div>
      <div class="video-frame-controls">
        <button type="button" data-step-frames="-1" data-fps="${escapeHtml(stream.fps)}">后退 1 帧</button>
        <span class="video-current-time">00:00.000</span>
        <button type="button" data-step-frames="1" data-fps="${escapeHtml(stream.fps)}">前进 1 帧</button>
      </div>
      <footer><button type="button" data-frame-review-toggle>切换到逐帧复核</button><span>${escapeHtml(stream.video_file)}</span></footer>
    </article>
  `).join("") : `<span class="empty">这次会话没有可播放的视频</span>`;
  for (const stream of streams) {
    frameReview.set(stream.camera_id, {
      active: false,
      frameIndex: 0,
      frames: Math.max(1, Number(stream.frames) || 1),
      fps: Math.max(1, Number(stream.fps) || 20),
    });
  }
  for (const video of videoGrid.querySelectorAll("video")) {
    video.addEventListener("timeupdate", () => {
      video.closest("article").querySelector(".video-current-time").textContent = formatTime(video.currentTime);
    });
    video.addEventListener("error", () => showReviewFrame(video.closest("article"), 0));
  }
  for (const button of videoGrid.querySelectorAll("[data-frame-review-toggle]")) {
    button.addEventListener("click", () => {
      const card = button.closest("article");
      const video = card.querySelector("video");
      const state = frameReview.get(card.dataset.cameraId);
      showReviewFrame(card, state.active ? state.frameIndex : video.currentTime * state.fps);
    });
  }
  for (const slider of videoGrid.querySelectorAll(".recorded-frame-review input")) {
    slider.addEventListener("input", () => showReviewFrame(slider.closest("article"), Number(slider.value)));
  }
  for (const button of videoGrid.querySelectorAll("[data-step-frames]")) {
    button.addEventListener("click", () => {
      const card = button.closest("article");
      const video = card.querySelector("video");
      const state = frameReview.get(card.dataset.cameraId);
      if (state.active) {
        showReviewFrame(card, state.frameIndex + Number(button.dataset.stepFrames));
        return;
      }
      const fps = state.fps;
      video.pause();
      video.currentTime = Math.max(
        0,
        video.currentTime + Number(button.dataset.stepFrames) / fps,
      );
    });
  }
  const cameraIds = streams.map((stream) => stream.camera_id);
  const options = cameraIds.map((cameraId) => `<option value="${escapeHtml(cameraId)}">${escapeHtml(cameraId)}</option>`).join("");
  sourceSelect.innerHTML = options;
  targetSelect.innerHTML = options;
  if (cameraIds.length > 1) targetSelect.selectedIndex = 1;
  resetMarkers();
}

function renderHandoffs(payload) {
  const entries = payload.annotations || [];
  handoffList.innerHTML = entries.length ? entries.map((item) => `
    <article class="handoff-row">
      <div><strong>${escapeHtml(item.from_camera)} → ${escapeHtml(item.to_camera)}</strong><span>${escapeHtml(item.person_id)} · ${describeGap(item.gap_seconds)}</span></div>
      <div><span>源：${formatTime(item.source_end.relative_seconds)} / 帧 ${escapeHtml(item.source_end.frame_index)}</span><span>目标：${formatTime(item.target_start.relative_seconds)} / 帧 ${escapeHtml(item.target_start.frame_index)}</span></div>
      <button type="button" data-delete-handoff="${escapeHtml(item.id)}">删除</button>
    </article>
  `).join("") : `<span class="empty">尚无标注</span>`;
  suggestions.innerHTML = Object.entries(payload.suggestions || {}).map(([pair, item]) => `
    <div><strong>${escapeHtml(pair)}</strong><span>${escapeHtml(item.samples)} 个样本 · 中位 ${describeGap(item.median_gap_seconds)} · P90 gap ${escapeHtml(item.p90_gap_seconds)} 秒</span></div>
  `).join("");
  for (const button of handoffList.querySelectorAll("[data-delete-handoff]")) {
    button.addEventListener("click", async () => {
      try {
        const payload = await api(sessionPath(`/handoffs/${button.dataset.deleteHandoff}`), {method: "DELETE"});
        renderHandoffs(payload);
      } catch (error) {
        showError(error.message);
      }
    });
  }
}

async function loadSession() {
  currentSession = sessions.find((item) => `${item.subject_id}/${item.session_id}` === sessionSelect.value);
  if (!currentSession) return;
  showError();
  renderVideos();
  try {
    renderHandoffs(await api(sessionPath("/handoffs")));
  } catch (error) {
    showError(error.message);
  }
}

async function initialize() {
  try {
    const payload = await api("/api/recordings");
    sessions = payload.sessions || [];
    sessionSelect.innerHTML = sessions.length ? sessions.map((item) => `
      <option value="${escapeHtml(`${item.subject_id}/${item.session_id}`)}">${escapeHtml(item.subject_id)} · ${escapeHtml(item.session_id)} · ${escapeHtml(item.elapsed_seconds)} 秒</option>
    `).join("") : `<option value="">尚无已保存录像</option>`;
    serviceState.textContent = "服务在线";
    serviceState.className = "service-state online";
    await loadSession();
  } catch (error) {
    serviceState.textContent = "服务断开";
    serviceState.className = "service-state error";
    showError(error.message);
  }
}

sessionSelect.addEventListener("change", loadSession);
sourceSelect.addEventListener("change", resetMarkers);
targetSelect.addEventListener("change", resetMarkers);
document.querySelector("#restartVideosButton").addEventListener("click", () => {
  for (const card of videoGrid.querySelectorAll(".recorded-video-card")) {
    const state = frameReview.get(card.dataset.cameraId);
    if (state?.active) showReviewFrame(card, 0);
    else {
      const video = card.querySelector("video");
      video.pause();
      video.currentTime = 0;
    }
  }
  resetMarkers();
});
document.querySelector("#pauseVideosButton").addEventListener("click", () => {
  for (const video of videoGrid.querySelectorAll("video")) video.pause();
});
document.querySelector("#captureSourceButton").addEventListener("click", () => {
  const video = videoFor(sourceSelect.value);
  sourceTime = selectedMediaTime(sourceSelect.value);
  if (sourceTime == null) return;
  video?.pause();
  sourceValue.textContent = `${sourceSelect.value} · ${formatTime(sourceTime)}`;
  updateGapPreview();
});
document.querySelector("#captureTargetButton").addEventListener("click", () => {
  const video = videoFor(targetSelect.value);
  targetTime = selectedMediaTime(targetSelect.value);
  if (targetTime == null) return;
  video?.pause();
  targetValue.textContent = `${targetSelect.value} · ${formatTime(targetTime)}`;
  updateGapPreview();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError();
  if (!form.reportValidity()) return;
  if (sourceSelect.value === targetSelect.value) return showError("源和目标摄像头不能相同");
  if (sourceTime == null || targetTime == null) return showError("请先标记源末帧和目标首帧");
  try {
    const payload = await api(sessionPath("/handoffs"), {
      method: "POST",
      body: JSON.stringify({
        person_id: personIdInput.value.trim(),
        from_camera: sourceSelect.value,
        to_camera: targetSelect.value,
        source_end_seconds: sourceTime,
        target_start_seconds: targetTime,
        notes: notesInput.value.trim(),
      }),
    });
    renderHandoffs(payload);
    resetMarkers();
  } catch (error) {
    showError(error.message);
  }
});

initialize();
