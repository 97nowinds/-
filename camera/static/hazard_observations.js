"use strict";

(() => {
  const card = document.querySelector("#visualHazardCard");
  if (!card) return;
  const summary = card.querySelector("strong");
  const status = card.querySelector("small");
  const cameraNames = {cam_1: "实验室 1", cam_2: "实验室 2", cam_entrance: "入口"};
  const kindNames = {smoke: "烟雾", flame: "火焰", spark: "火星"};
  async function refresh() {
    try {
      const response = await fetch("/api/hazards/observations", {cache: "no-store"});
      if (!response.ok) throw new Error();
      const report = await response.json();
      const cameras = Object.values(report.cameras || {});
      card.dataset.state = report.state;
      if (!cameras.length) {
        card.dataset.warning = "";
        summary.textContent = "等待观察数据";
        status.textContent = report.state === "stale" ? "上次观察已经过期" : "烟雾与火星";
        return;
      }
      const recent = report.recent?.[0];
      card.dataset.warning = recent?.kind || "";
      if (recent) {
        const time = new Date(recent.at * 1000).toLocaleTimeString("zh-CN", {hour12:false});
        const score = recent.score === null ? "" : ` · ${Math.round(recent.score * 100)} 分`;
        const title = recent.stage === "visual_ongoing" ? "火焰持续预警" : `${kindNames[recent.kind]}预警`;
        summary.textContent = `${title} · 待复核`;
        status.textContent = `${cameraNames[recent.camera]} · ${time}${score}`;
      } else {
        summary.textContent = `${cameras.length}/3 路观察在线`;
        status.textContent = "最近 2 分钟无视觉预警";
      }
    } catch {
      card.dataset.state = "unavailable";
      card.dataset.warning = "";
      summary.textContent = "观察数据不可用";
      status.textContent = "烟雾与火星";
    } finally {
      window.setTimeout(refresh, 3000);
    }
  }
  void refresh();
})();
