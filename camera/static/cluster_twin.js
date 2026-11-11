const ClusterTwin = (() => {
  const endpoint = "/api/twin/frame";
  const overview = {yaw:.57, pitch:.82, zoom:1};
  const frameInterval = 120;
  let root, image, hotspots, status, view, scene, drag, timer, imageUrl;
  let pending = false;
  let visible = false;
  let displayed = "";
  let nextRequest = 0;

  function schedule() {
    if (!visible || pending || timer || !scene) return;
    timer = setTimeout(requestFrame, Math.max(0,nextRequest-performance.now()));
  }

  async function requestFrame() {
    timer = null;
    if (!visible) return;
    const body = JSON.stringify({...scene, view});
    if (body === displayed) return;
    pending = true;
    nextRequest = performance.now() + frameInterval;
    try {
      const response = await fetch(endpoint, {
        method:"POST", headers:{"Content-Type":"application/json"}, body,
        signal:AbortSignal.timeout(4500), cache:"no-store",
      });
      if (!response.ok || !response.headers.get("Content-Type")?.startsWith("image/jpeg")) {
        throw new Error("Cluster frame unavailable");
      }
      const url = URL.createObjectURL(await response.blob());
      const bounds = JSON.parse(response.headers.get("X-Instrument-Bounds") || "[]");
      updateHotspots(bounds);
      const old = imageUrl;
      image.src = imageUrl = url;
      if (old) URL.revokeObjectURL(old);
      displayed = body;
      image.hidden = false;
      status.textContent = "集群渲染 · 拖动旋转 · 滚轮缩放";
    } catch {
      displayed = "";
      image.hidden = true;
      hotspots.replaceChildren();
      status.textContent = "集群画面暂不可用 · 正在重连";
      nextRequest = performance.now() + 1500;
    } finally {
      pending = false;
      if (!displayed || body !== JSON.stringify({...scene, view})) schedule();
    }
  }

  function updateHotspots(bounds) {
    const names = new Map(scene.equipment.map(item => [item.instrument_id, item.name]));
    const existing = new Map(Array.from(hotspots.children, button => [button.dataset.id, button]));
    for (const [id,x,y,width,height] of bounds) {
      let button = existing.get(id);
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.dataset.id = id;
        button.className = "cluster-twin-hotspot";
        button.addEventListener("click", () => {
          view = {yaw:.38,pitch:.48,zoom:.17,instrument:id};
          schedule();
        });
        hotspots.append(button);
      }
      existing.delete(id);
      button.title = `查看${names.get(id) || id}三维模型`;
      button.setAttribute("aria-label",button.title);
      Object.assign(button.style,{
        left:`${x/10}%`,top:`${y/6.4}%`,width:`${width/10}%`,height:`${height/6.4}%`,
      });
    }
    for (const button of existing.values()) button.remove();
  }

  function create(host) {
    root = document.createElement("div");
    root.className = "twin-webgl";
    image = document.createElement("img");
    image.alt = "集群实时绘制的实验室三维场景";
    image.draggable = false;
    image.hidden = true;
    const frame = document.createElement("div");
    frame.className = "cluster-twin-frame";
    hotspots = document.createElement("div");
    hotspots.className = "cluster-twin-hotspots";
    frame.append(image,hotspots);
    status = document.createElement("div");
    status.className = "twin-stage-hint";
    status.textContent = "正在连接集群画面";
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "tool-link twin-reset";
    reset.textContent = "重置视角";
    view = {...overview};
    reset.addEventListener("click", () => { view = {...overview}; schedule(); });
    image.addEventListener("pointerdown", event => {
      if (event.button !== 0 || !event.isPrimary) return;
      drag = [event.clientX,event.clientY];
      image.setPointerCapture(event.pointerId);
    });
    image.addEventListener("pointermove", event => {
      if (!drag || !event.isPrimary) return;
      view.yaw = (view.yaw - (event.clientX-drag[0])*.009) % (2*Math.PI);
      view.pitch = Math.max(.18,Math.min(1.48,view.pitch+(event.clientY-drag[1])*.009));
      drag = [event.clientX,event.clientY];
      schedule();
    });
    image.addEventListener("lostpointercapture", () => { drag = null; });
    image.addEventListener("wheel", event => {
      event.preventDefault();
      view.zoom = Math.max(.16,Math.min(2.2,view.zoom*Math.exp(event.deltaY*.001)));
      schedule();
    }, {passive:false});
    root.classList.add("cluster-twin");
    root.append(frame,status,reset);
    host.append(root);
  }

  function render(host, map, equipment, personLabel, trails) {
    if (!root) create(host);
    visible = true;
    root.hidden = false;
    const people = (map.people || []).filter(person => person.map_visible !== false &&
      [person.x,person.y].every(Number.isFinite)).map(person => ({
        x:person.x, y:person.y, identified:person.identified,
        track_id:person.track_id, label:personLabel(person),
      }));
    scene = {
      map:{width_m:map.width_m,height_m:map.height_m,fixtures:map.fixtures || [],
        cameras:map.cameras || {},people},
      equipment,
      trails:people.map(person => [person.track_id,(trails.get(person.track_id) || []).slice(-16)]),
    };
    schedule();
    return !image.hidden;
  }

  function hide() {
    visible = false;
    clearTimeout(timer);
    timer = null;
    if (root) root.hidden = true;
  }

  return {render,hide};
})();
