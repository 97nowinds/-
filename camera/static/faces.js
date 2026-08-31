const API_BASE = (window.__LAB_API_BASE__ || new URLSearchParams(location.search).get("api") || location.origin).replace(/\/$/, "");
const monitorLink = document.querySelector("#monitorLink");
if (monitorLink) monitorLink.href = `/?api=${encodeURIComponent(API_BASE)}`;
const form = document.querySelector("#registrationForm");
const cameraId = document.querySelector("#cameraId");
const targetSamples = document.querySelector("#targetSamples");
const replaceExisting = document.querySelector("#replaceExisting");
const startSession = document.querySelector("#startSession");
const cancelSession = document.querySelector("#cancelSession");
const captureButton = document.querySelector("#captureButton");
const finishButton = document.querySelector("#finishButton");
const captureStream = document.querySelector("#captureStream");
const previewMessage = document.querySelector("#previewMessage");
const liveGuidance = document.querySelector("#liveGuidance");
const sessionState = document.querySelector("#sessionState");
const captureCount = document.querySelector("#captureCount");
const poseLabel = document.querySelector("#poseLabel");
const poseInstruction = document.querySelector("#poseInstruction");
const poseProgress = document.querySelector("#poseProgress");
const captureMessage = document.querySelector("#captureMessage");
const faceServiceState = document.querySelector("#faceServiceState");
let session = null;
let previewPending = false;
let camerasLoaded = false;

function setMessage(message, error = false) {
  captureMessage.textContent = message;
  captureMessage.className = `capture-message${error ? " error" : ""}`;
}

function updateSession(payload) {
  session = payload;
  const active = Boolean(payload?.active);
  if (!active) {
    sessionState.textContent = "尚未开始";
    captureCount.textContent = `0 / ${targetSamples.value}`;
    poseLabel.textContent = "等待开始";
    poseInstruction.textContent = "填写资料并选择入口摄像头，然后点击“开始采集会话”";
    poseProgress.textContent = "当前姿态 0 / 0";
    setPreviewStream(cameraId.value);
    previewMessage.hidden = false;
    liveGuidance.textContent = "等待实时检测";
    liveGuidance.className = "live-guidance idle";
    captureButton.disabled = true;
    finishButton.disabled = true;
    cancelSession.disabled = true;
    return;
  }
  const pose = payload.pose || {};
  sessionState.textContent = `采集中 · ${payload.name}`;
  captureCount.textContent = `${payload.sample_count} / ${payload.target_samples}`;
  poseLabel.textContent = `当前姿态：${pose.label || ""}`;
  poseInstruction.textContent = pose.instruction || "按当前姿态保持稳定";
  poseProgress.textContent = `当前姿态 ${payload.pose_count || 0} / ${pose.target || 0} · 阶段 ${Number(payload.pose_index || 0) + 1} / ${payload.pose_total || 0}`;
  captureButton.disabled = false;
  finishButton.disabled = !payload.ready;
  cancelSession.disabled = false;
  previewMessage.hidden = true;
  setPreviewStream(payload.camera_id);
  if (payload.last_result?.message) setMessage(payload.last_result.message, !payload.last_result.accepted);
  else setMessage(`当前姿态：${pose.label || "请按提示调整"}。${pose.instruction || "保持稳定后点击“采集一张”"}`);
}

function renderPreview(preview) {
  if (!cameraId.value) return;
  let message = preview.message || "正在检测人脸";
  if (preview.reason === "face_not_found") {
    message = "已检测到人体，但没有清晰人脸：请正对镜头、抬头并靠近一些";
  } else if (preview.reason === "no_person") {
    message = "未检测到人体：请站到摄像头前并保持单人入镜";
  } else if (preview.ready) {
    message = "已检测到清晰人脸，可以点击“采集一张”";
  }
  liveGuidance.textContent = message;
  liveGuidance.className = `live-guidance ${preview.ready ? "ready" : ""}`;
  const quality = preview.quality;
  if (quality) {
    const details = `尺寸 ${quality.width}×${quality.height} · 亮度 ${quality.brightness} · 清晰度 ${quality.sharpness}`;
    liveGuidance.textContent += `｜${details}`;
  }
}

async function refreshPreview() {
  if (!previewPending && cameraId.value) {
    previewPending = true;
    try { renderPreview(await request(`/api/face-registration/preview?camera_id=${encodeURIComponent(cameraId.value)}`)); }
    catch (error) {
      liveGuidance.textContent = `实时检测失败：${error.message}`;
      liveGuidance.className = "live-guidance";
    } finally { previewPending = false; }
  }
  window.setTimeout(refreshPreview, 450);
}

function setPreviewStream(selectedCamera) {
  if (!selectedCamera) return;
  captureStream.dataset.cameraId = selectedCamera;
  captureStream.src = `${API_BASE}/video/${encodeURIComponent(selectedCamera)}?registration=1&ts=${Date.now()}`;
  previewMessage.hidden = true;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {cache: "no-store", ...options});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

async function loadCameras() {
  try {
    const state = await request("/api/face-registration");
    const previousCamera = cameraId.value;
    cameraId.innerHTML = state.cameras.map((camera) => `<option value="${camera.id}">${camera.name} · ${camera.status}</option>`).join("");
    const rtspCamera = state.cameras.find((camera) => camera.role === "entrance_identity" && camera.status === "running")
      || state.cameras.find((camera) => camera.source_type === "rtsp" && camera.status === "running")
      || state.cameras.find((camera) => camera.source_type === "rtsp");
    cameraId.value = state.cameras.some((camera) => camera.id === previousCamera)
      ? previousCamera
      : (rtspCamera?.id || state.cameras[0]?.id || "");
    if (!camerasLoaded || captureStream.dataset.cameraId !== cameraId.value) setPreviewStream(cameraId.value);
    camerasLoaded = true;
    // Restore a session that may have been created in another browser tab.
    if (state.active) updateSession(state);
    faceServiceState.textContent = "服务在线";
    faceServiceState.className = "service-state online";
  } catch (error) {
    camerasLoaded = false;
    faceServiceState.textContent = "服务断开";
    faceServiceState.className = "service-state error";
    setMessage(`等待后端服务：${error.message}。输入海康密码后页面会自动恢复。`, true);
  } finally {
    if (!camerasLoaded) window.setTimeout(loadCameras, 1000);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const personId = document.querySelector("#personId").value.trim();
  const personName = document.querySelector("#personName").value.trim();
  const personLabel = document.querySelector("#personLabel").value.trim();
  const selectedCamera = cameraId.value;
  if (!personId || !personName || !personLabel || !selectedCamera) {
    setMessage("请先填写人员 ID、姓名、注册序号并选择采集摄像头。", true);
    return;
  }
  startSession.disabled = true;
  try {
    const result = await request("/api/face-registration/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        person_id: personId,
        name: personName,
        label: personLabel,
        camera_id: selectedCamera,
        target_samples: targetSamples.value,
        replace_existing: replaceExisting.checked,
      }),
    });
    updateSession(result);
    setMessage(`会话已建立：第 1 步【${result.pose.label}】。${result.pose.instruction}，画面稳定后点击“采集一张”。`);
  } catch (error) {
    sessionState.textContent = "建立失败";
    setMessage(`无法开始采集：${error.message}`, true);
  }
  finally { startSession.disabled = false; }
});

cameraId.addEventListener("change", () => {
  setPreviewStream(cameraId.value);
  if (!session?.active) {
    liveGuidance.textContent = "正在检查实时画面";
    liveGuidance.className = "live-guidance idle";
  }
});

captureButton.addEventListener("click", async () => {
  captureButton.disabled = true;
  try { updateSession(await request("/api/face-registration/capture", {method: "POST"})); }
  catch (error) { setMessage(error.message, true); }
  finally { if (session?.active) captureButton.disabled = false; }
});

finishButton.addEventListener("click", async () => {
  finishButton.disabled = true;
  try {
    const result = await request("/api/face-registration/finalize", {method: "POST"});
    updateSession({active: false});
    setMessage(`注册完成：${result.name}，共保存 ${result.sample_count} 张有效样本。`);
  } catch (error) { setMessage(error.message, true); finishButton.disabled = false; }
});

cancelSession.addEventListener("click", async () => {
  try { await request("/api/face-registration/cancel", {method: "POST"}); updateSession({active: false}); setMessage("本次采集已取消，临时样本已删除。", true); }
  catch (error) { setMessage(error.message, true); }
});

loadCameras();
refreshPreview();
