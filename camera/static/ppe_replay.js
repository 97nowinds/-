(() => {
  if (document.body.classList.contains('presentation-blue')) return;
  const section = document.createElement('section');
  section.id = 'ppeReplayTest';
  section.style.cssText = 'margin:20px 0;padding:20px;border:2px solid #eab308;border-radius:12px;background:#11283c;color:white';
  const title = document.createElement('h2');
  title.textContent = 'PPE录像回放测试 · 非现场报警';
  const detail = document.createElement('p');
  section.append(title, detail);
  (document.querySelector('main') || document.body).append(section);
  async function refresh() {
    try {
      const response = await fetch('/api/ppe/replay-test', {cache:'no-store'});
      if (!response.ok) throw new Error('unavailable');
      const data = await response.json();
      if (data.test_mode === 'not_run') { detail.textContent = '暂无录像回放测试记录'; return; }
      const sent = data.delivery.filter(item => item.state === 'sent').length;
      detail.textContent = `录像：${data.source_session}。纯模型产生告警 ${data.model_only_alerts} 条；人工真值测试分支产生 ${data.manual_truth_test_alerts} 条，飞书接口确认投递 ${sent} 条。人工真值分支只验证告警链路，不代表模型识别通过。`;
      if (!section.querySelector('img')) {
        const image = document.createElement('img');
        image.src = '/api/ppe/replay-test/evidence.jpg';
        image.alt = '录像回放测试证据：人工真值测试分支，非当前现场违规';
        image.style.cssText = 'max-width:640px;width:100%;display:block';
        section.append(image);
      }
    } catch { detail.textContent = '回放测试记录暂不可用，不代表现场状态'; }
  }
  refresh();
})();
