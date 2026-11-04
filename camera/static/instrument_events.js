(() => {
  const section = document.createElement('section');
  section.id = 'instrumentHistory';
  section.style.cssText = 'margin:20px 0;padding:20px;background:#11283c;color:#e7eef6;border:1px solid #29465e;border-radius:12px;overflow:auto';
  const title = document.createElement('h2'); title.textContent = '仪器使用事件记录';
  const status = document.createElement('p');
  const table = document.createElement('table'); table.style.cssText = 'width:100%;border-collapse:collapse;white-space:nowrap;font-size:14px';
  section.append(title, status, table);
  (document.querySelector('main') || document.body).append(section);
  const time = at => at ? new Date(at).toLocaleString('zh-CN', {hour12:false,timeZone:'Asia/Shanghai'}) : '等待有效结束判断';
  function cell(row, text, tag='td') {
    const c = document.createElement(tag); c.textContent = text; c.style.cssText='padding:10px;text-align:left;border-bottom:1px solid #29465e'; row.append(c);
  }
  async function refresh() {
    try {
      const data = await LivePolling.fetchJSON('/api/instrument-events');
      window.dispatchEvent(new CustomEvent('lab-instrument-history', {detail:data}));
      const fresh = data.sync.state === 'connected' && Date.now()-Date.parse(data.sync.last_success_at)<20000;
      status.textContent = `${fresh ? '记录同步正常' : '记录同步中断，保留已保存记录'} · 最近同步 ${data.sync.last_success_at ? time(data.sync.last_success_at) : '尚未成功'} · 时间依据：源画面`;
      table.replaceChildren();
      const header = document.createElement('tr');
      for (const text of ['人员 / 轨迹','摄像头','仪器','开始（北京时间）','结束（北京时间）','时长','状态']) cell(header,text,'th');
      table.append(header);
      for (const s of data.sessions.slice(0,20)) {
        const row=document.createElement('tr');
        for (const value of [s.person_label,s.camera_id,s.instrument_name,time(s.started_at),time(s.ended_at),
          s.duration_seconds == null ? '尚未结束' : `${s.duration_seconds.toFixed(3)} 秒`,s.status==='ended'?'已结束':s.status==='interrupted'?'中断／结束未确认：'+s.interruption_reason:'未取得结束判断']) cell(row,value);
        table.append(row);
      }
      if (!data.sessions.length) { const row=document.createElement('tr'); cell(row,'暂无有效仪器使用事件'); table.append(row); }
    } catch {
      status.textContent = '记录接口连接中断；保留上次显示，不据此结束操作';
    }
    setTimeout(refresh,5000);
  }
  refresh();
})();
