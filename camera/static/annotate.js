(function () {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const API_BASE = (params.get("api") || window.__LAB_API_BASE__ || "http://127.0.0.1:5000").replace(/\/$/, "");
  const cameraSelect = document.getElementById("cameraSelect");
  const modeSelect = document.getElementById("modeSelect");
  const frame = document.getElementById("cameraFrame");
  const canvas = document.getElementById("annotationCanvas");
  const ctx = canvas.getContext("2d");
  const mapSvg = document.getElementById("mapSvg");
  const mapPointsLayer = document.getElementById("mapPoints");
  const mapStep = document.getElementById("mapStep");
  const pointStep = document.getElementById("pointStep");
  const pointList = document.getElementById("pointList");
  const status = document.getElementById("status");
  const state = { config: null, annotations: {}, cameraId: null, draft: null, drag: null, pendingImage: null };
  const colors = { main_aisle: "#4bb4ff", secondary_aisle: "#b18cff", overlap: "#f0ae5a" };

  function setStatus(message, isError) {
    status.textContent = message;
    status.classList.toggle("error", Boolean(isError));
  }

  function clone(value) { return JSON.parse(JSON.stringify(value)); }

  function emptyDraft(cameraId) {
    const camera = state.config.cameras[cameraId] || {};
    const saved = state.annotations[cameraId] || {};
    return {
      regions: clone(saved.regions || {}),
      image_points: clone(saved.image_points || camera.image_points || []),
      map_points: clone(saved.map_points || camera.map_points || []),
    };
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawCanvas();
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) };
  }

  function drawCanvas() {
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 1;
    const height = rect.height || 1;
    ctx.clearRect(0, 0, width, height);
    if (!state.draft) return;
    Object.entries(state.draft.regions || {}).forEach(([type, region]) => {
      ctx.fillStyle = `${colors[type] || "#fff"}24`;
      ctx.strokeStyle = colors[type] || "#fff";
      ctx.lineWidth = 2;
      ctx.setLineDash(type === "overlap" ? [8, 5] : []);
      ctx.fillRect(region.x * width, region.y * height, region.width * width, region.height * height);
      ctx.strokeRect(region.x * width, region.y * height, region.width * width, region.height * height);
      ctx.font = "600 13px Microsoft YaHei, sans-serif";
      ctx.fillStyle = colors[type] || "#fff";
      ctx.fillText({ main_aisle: "主通道", secondary_aisle: "内侧通道", overlap: "重叠区" }[type] || type, region.x * width + 8, region.y * height + 18);
    });
    ctx.setLineDash([]);
    (state.draft.image_points || []).forEach((point, index) => {
      const x = point[0] * width; const y = point[1] * height;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fillStyle = "#70d89a"; ctx.fill();
      ctx.strokeStyle = "#102017"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "700 12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(index + 1), x, y);
    });
    ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
  }

  function svgElement(tag, attrs, text) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs || {}).forEach(([key, value]) => element.setAttribute(key, value));
    if (text) element.textContent = text;
    return element;
  }

  function mapCoordinate(event) {
    const rect = mapSvg.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width * 1200, y: (event.clientY - rect.top) / rect.height * 700 };
  }

  function drawMap() {
    if (!state.config) return;
    ["mapFixtures", "mapZones", "mapCameras"].forEach((id) => { document.getElementById(id).replaceChildren(); });
    const sx = 1200 / state.config.width_m; const sy = 700 / state.config.height_m;
    (state.config.fixtures || []).forEach((fixture) => {
      const x = fixture.x * sx; const y = fixture.y * sy; const w = fixture.width * sx; const h = fixture.height * sy;
      document.getElementById("mapFixtures").append(svgElement("rect", { class: "map-fixture", x, y, width: w, height: h, rx: 3 }), svgElement("text", { class: "map-label", x: x + w / 2, y: y + h / 2 + 6 }, fixture.name));
    });
    (state.config.zones || []).forEach((zone) => {
      const x = zone.x * sx; const y = zone.y * sy; const w = zone.width * sx; const h = zone.height * sy;
      document.getElementById("mapZones").append(svgElement("rect", { class: `map-zone ${zone.kind === "overlap" ? "overlap" : ""}`, x, y, width: w, height: h }), svgElement("text", { class: "map-label", x: x + w / 2, y: y + h / 2 + 5 }, zone.name));
    });
    Object.entries(state.config.cameras || {}).forEach(([id, camera]) => {
      const x = camera.position[0] * sx; const y = camera.position[1] * sy;
      document.getElementById("mapCameras").append(svgElement("circle", { class: "map-camera", cx: x, cy: y, r: 9 }), svgElement("text", { class: "map-label", x, y: y - 15 }, camera.label || id));
    });
    mapPointsLayer.replaceChildren();
    (state.draft && state.draft.map_points || []).forEach((point, index) => {
      const x = point[0] / state.config.width_m * 1200; const y = point[1] / state.config.height_m * 700;
      mapPointsLayer.append(svgElement("circle", { class: "map-point", cx: x, cy: y, r: 10 }), svgElement("text", { class: "map-point-label", x, y: y + 5 }, String(index + 1)));
    });
    pointList.textContent = state.draft && state.draft.map_points.length ? `已配对 ${state.draft.map_points.length}/4 个地面点` : "尚未设置参考点";
  }

  function selectCamera(cameraId) {
    state.cameraId = cameraId;
    state.draft = emptyDraft(cameraId);
    state.drag = null; state.pendingImage = null;
    frame.src = `${API_BASE}/video/${encodeURIComponent(cameraId)}?remote=1`;
    document.getElementById("cameraTitle").textContent = state.config.cameras[cameraId].label || cameraId;
    setStatus(`当前为 ${cameraId}，可以开始标注。`, false);
    drawMap(); drawCanvas(); updateSteps();
  }

  function updateSteps() {
    const reference = modeSelect.value === "reference";
    pointStep.textContent = reference ? `画面点 ${state.draft ? state.draft.image_points.length : 0}/4` : "区域标注";
    mapStep.textContent = state.pendingImage === null ? "等待标注" : `请点击地图设置点 ${state.pendingImage + 1}`;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.draft) return;
    const point = canvasPoint(event);
    if (modeSelect.value === "reference") {
      if (state.pendingImage !== null) return;
      if (state.draft.image_points.length >= 4) { setStatus("四个画面点已完成；请清除当前标注后重来。", true); return; }
      state.draft.image_points.push([Number(point.x.toFixed(5)), Number(point.y.toFixed(5))]);
      state.pendingImage = state.draft.image_points.length - 1;
      setStatus(`已选择画面点 ${state.pendingImage + 1}，请在右侧地图点击对应位置。`, false);
      drawCanvas(); drawMap(); updateSteps();
      return;
    }
    state.drag = { start: point, current: point };
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    state.drag.current = canvasPoint(event); drawCanvas();
    const rect = canvas.getBoundingClientRect(); const a = state.drag.start; const b = state.drag.current;
    ctx.strokeStyle = colors[modeSelect.value]; ctx.setLineDash([8, 5]); ctx.strokeRect(Math.min(a.x, b.x) * rect.width, Math.min(a.y, b.y) * rect.height, Math.abs(a.x - b.x) * rect.width, Math.abs(a.y - b.y) * rect.height); ctx.setLineDash([]);
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!state.drag) return;
    const a = state.drag.start; const b = state.drag.current; state.drag = null;
    const region = { x: Number(Math.min(a.x, b.x).toFixed(5)), y: Number(Math.min(a.y, b.y).toFixed(5)), width: Number(Math.abs(a.x - b.x).toFixed(5)), height: Number(Math.abs(a.y - b.y).toFixed(5)) };
    if (region.width < 0.02 || region.height < 0.02) { setStatus("框选区域太小，请重新拖拽。", true); drawCanvas(); return; }
    state.draft.regions[modeSelect.value] = region; setStatus("区域已记录，可以继续标注或保存。", false); drawCanvas();
  });

  mapSvg.addEventListener("click", (event) => {
    if (state.pendingImage === null || !state.draft) return;
    const point = mapCoordinate(event); const index = state.pendingImage;
    state.draft.map_points[index] = [Number((point.x / 1200 * state.config.width_m).toFixed(3)), Number((point.y / 700 * state.config.height_m).toFixed(3))];
    state.pendingImage = null;
    setStatus(`参考点 ${index + 1} 已配对，请继续点击画面设置下一个点。`, false); drawMap(); updateSteps();
  });

  modeSelect.addEventListener("change", () => { state.drag = null; setStatus(modeSelect.value === "reference" ? "按顺序点击画面点，再点击地图对应位置。" : "在画面中拖拽框选区域。", false); updateSteps(); drawCanvas(); });
  cameraSelect.addEventListener("change", () => selectCamera(cameraSelect.value));
  window.addEventListener("resize", resizeCanvas);
  document.getElementById("clearButton").addEventListener("click", () => {
    if (!state.draft) return;
    if (modeSelect.value === "reference") { state.draft.image_points = []; state.draft.map_points = []; state.pendingImage = null; }
    else delete state.draft.regions[modeSelect.value];
    setStatus("当前标注已清除。", false); drawCanvas(); drawMap(); updateSteps();
  });
  document.getElementById("saveButton").addEventListener("click", async () => {
    if (!state.draft || state.pendingImage !== null) { setStatus("请先完成当前参考点配对。", true); return; }
    try {
      const response = await fetch(`${API_BASE}/api/annotation/save`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ camera_id: state.cameraId, regions: state.draft.regions, image_points: state.draft.image_points, map_points: state.draft.map_points, activate_calibration: document.getElementById("activateCalibration").checked }) });
      if (response.status === 404) throw new Error("当前后端未加载标注保存接口，请重启后端启动器后再保存");
      const result = await response.json(); if (!response.ok || !result.ok) throw new Error(result.error || "保存失败");
      state.annotations[state.cameraId] = clone(state.draft); setStatus(`${result.message} 当前摄像头：${state.cameraId}`, false);
    } catch (error) { setStatus(error.message, true); }
  });

  async function boot() {
    try {
      let response = await fetch(`${API_BASE}/api/annotation/config`);
      if (!response.ok) {
        // Older backend processes can still provide the map through /api/state.
        // This keeps the annotation preview usable until that process is restarted.
        response = await fetch(`${API_BASE}/api/state`);
        if (!response.ok) throw new Error(`配置读取失败（${response.status}）`);
        const legacy = await response.json();
        state.config = { ...(legacy.floor_map || {}), cameras: {} };
        (legacy.cameras || []).forEach((camera) => {
          state.config.cameras[camera.id] = { label: camera.name || camera.id, position: [0, 0], image_points: [], map_points: [] };
        });
        state.annotations = {};
        setStatus("当前后端尚未重启：已用现有地图预览，保存前请重启后端。", false);
      } else {
        state.config = await response.json(); state.annotations = state.config.annotations || {};
      }
      Object.entries(state.config.cameras || {}).forEach(([id, camera]) => cameraSelect.append(new Option(camera.label || id, id)));
      if (!cameraSelect.options.length) throw new Error("没有可用摄像头");
      selectCamera(cameraSelect.options[0].value); frame.addEventListener("load", resizeCanvas); resizeCanvas();
    } catch (error) { setStatus(error.message, true); }
  }
  boot();
})();
