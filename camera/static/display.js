(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const api = (window.__LAB_API_BASE__ || new URLSearchParams(location.search).get('api') || location.origin).replace(/\/$/, '');
  $('maintenance').href = `/monitor?api=${encodeURIComponent(api)}`;
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = (id, value) => { $(id).textContent = value; };
  const clock = () => text('clock', new Date().toLocaleString('zh-CN', {hour12:false}));
  clock(); setInterval(clock, 1000);
  $('fullscreen').onclick = async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { text('fullscreen','全屏不可用'); } };
  document.addEventListener('fullscreenchange', () => text('fullscreen', document.fullscreenElement ? '退出全屏' : '全屏展示'));
  async function get(url) {
    const response = await fetch(url, {cache:'no-store', signal:AbortSignal.timeout(4500)});
    if (!response.ok) throw new Error('数据不可用');
    return response.json();
  }
  function svg(tag, attrs, label) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
    if (label != null) el.textContent = label;
    $('map').append(el); return el;
  }
  function map(data) {
    $('map').replaceChildren();
    const valid = data && Number(data.width_m)>0 && Number(data.height_m)>0;
    $('mapEmpty').hidden = Boolean(valid);
    // Explicit display prevents the empty overlay's CSS grid from overriding hidden.
    $('mapEmpty').style.display = valid ? 'none' : 'grid';
    if (!valid) { text('mapMode','等待位置数据'); text('mapOccupancy','等待人员位置'); return; }
    text('mapMode', data.calibrated ? '实时位置图' : '空间示意 · 位置仅供参考');
    const x = n => 40 + Number(n) / data.width_m * 920;
    const y = n => 40 + Number(n) / data.height_m * 520;
    const rect = item => ({x:x(item.x), y:y(item.y), width:Number(item.width)/data.width_m*920, height:Number(item.height)/data.height_m*520});
    const usable = item => [item.x,item.y,item.width,item.height].every(n => n != null && Number.isFinite(Number(n))) && item.width>0 && item.height>0;
    svg('rect',{x:40,y:40,width:920,height:520,rx:18,class:'space-room'});
    // Render circulation first; physical furniture defines work areas without duplicate zone outlines.
    for (const item of (data.zones || []).filter(z => /aisle|service/.test(z.id) && usable(z))) {
      const r=rect(item);
      svg('rect',{...r,rx:8,class:'space-aisle'});
      svg('text',{x:r.x+r.width/2,y:r.y+r.height/2+5,'text-anchor':'middle',class:'space-aisle-label'},item.name || '通道');
    }
    const fixtures=(data.fixtures || []).filter(usable);
    for (const item of fixtures) {
      const r=rect(item);
      if (item.type === 'door') {
        svg('rect',{...r,rx:7,class:'space-door'});
        svg('text',{x:r.x+r.width/2,y:r.y+r.height/2-5,'text-anchor':'middle',class:'space-door-icon'},'↗');
        svg('text',{x:r.x+r.width/2,y:r.y+r.height/2+17,'text-anchor':'middle',class:'space-door-label'},'入口');
        continue;
      }
      svg('rect',{...r, y:r.y+4,rx:10,class:'space-fixture-shadow'});
      svg('rect',{...r,rx:10,class:`space-fixture ${item.type === 'bench' ? 'space-bench' : ''}`});
      svg('line',{x1:r.x+16,y1:r.y+10,x2:r.x+r.width-16,y2:r.y+10,class:'space-fixture-edge'});
      svg('text',{x:r.x+r.width/2,y:r.y+r.height/2+5,'text-anchor':'middle',class:'space-fixture-label'},item.name || '工作区域');
      if (r.width>250) {
        for (let offset=80;offset<r.width-35;offset+=90) svg('line',{x1:r.x+offset,y1:r.y+20,x2:r.x+offset,y2:r.y+r.height-18,class:'space-divider'});
      }
    }
    if (!fixtures.length) {
      for (const item of (data.zones || []).filter(z => z.kind !== 'overlap' && z.id !== 'overlap' && !/aisle|service/.test(z.id) && usable(z))) {
        const r=rect(item);
        svg('rect',{...r,rx:10,class:'space-fixture'});
        svg('text',{x:r.x+r.width/2,y:r.y+r.height/2+5,'text-anchor':'middle',class:'space-fixture-label'},item.name || '工作区域');
      }
    }
    const visible=(data.people || []).filter(p => p.map_visible !== false && p.x != null && p.y != null && Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y)));
    visible.forEach((p,index) => {
      const px=x(Math.max(0,Math.min(data.width_m,Number(p.x))));
      const py=y(Math.max(0,Math.min(data.height_m,Number(p.y))));
      svg('circle',{cx:px,cy:py,r:23,class:'space-person-halo'});
      const dot=svg('circle',{cx:px,cy:py,r:13,class:'space-person'});
      const title=document.createElementNS('http://www.w3.org/2000/svg','title');
      title.textContent=p.name || p.display_name || p.track_id || '匿名人员'; dot.append(title);
      svg('text',{x:px,y:py+4,'text-anchor':'middle',class:'space-person-number'},index+1);
      const label=String(p.name || p.display_name || p.track_id || '匿名人员');
      const width=Math.min(160,Math.max(68,label.length*12+22));
      const lx=Math.max(42,Math.min(958-width,px-width/2));
      const ly=py>515 ? py-49 : py+25;
      svg('rect',{x:lx,y:ly,width,height:26,rx:7,class:'space-person-tag'});
      svg('text',{x:lx+width/2,y:ly+18,'text-anchor':'middle',class:'space-person-label'},label.length>11 ? label.slice(0,10)+'…' : label);
    });
    text('mapOccupancy', visible.length ? `${visible.length} 人位置可见` : '当前暂无可见人员');
  }

  function environment(e) {
    const fresh = e && e.last_seen_at && ['online','alarm','running','connected','ok'].includes(e.status) && (e.age_seconds == null || e.age_seconds <= 15);
    const number = n => n != null && Number.isFinite(Number(n)) ? Number(n).toFixed(1) : '—';
    text('temperature', fresh && e.dht_valid ? number(e.temperature) : '—');
    text('humidity', fresh && e.dht_valid ? number(e.humidity) : '—');
    const alarm = fresh && e.flame_alarm;
    text('safety', !fresh ? '待接入' : alarm ? '火情告警' : '已接入');
    text('safetyNote', !fresh ? '传感器数据不可用或已过期' : alarm ? '请及时核查现场' : '环境传感器持续更新');
    text('flame', !fresh ? '火情状态 · 等待有效数据' : alarm ? '火情状态 · 检测到告警' : '火情状态 · 未检测到告警');
    $('flame').classList.toggle('alarm',Boolean(alarm)); $('safety').classList.toggle('alarm',Boolean(alarm));
  }
  function state(data) {
    const people = data.floor_map?.people || [];
    text('peopleCount', data.floor_map ? people.length : '—');
    text('identityCount', data.floor_map ? new Set(people.filter(p => p.person_id).map(p => p.person_id)).size : '—');
    $('people').innerHTML = people.length ? people.map(p => `<div class="person"><span class="avatar">◎</span><div>${escape(p.name || p.display_name || p.track_id || '匿名人员')}<small>${escape(p.zone || '区域待确认')}${p.map_visible === false ? ' · 位置待确认' : ''}</small></div></div>`).join('') : '<p class="empty">当前暂无人员位置数据</p>';
    map(data.floor_map);
    text('updated', `最近更新 ${new Date().toLocaleTimeString('zh-CN',{hour12:false})}`);
  }
  function results(payload) {
    const contract = window.__LAB_RESULT_CONTRACT__;
    if (contract && payload?.result_contract !== contract) throw new Error('结果协议不匹配');
    const rows = (Array.isArray(payload) ? payload : payload.results || []).filter(r => {
      if (r.status && r.status !== 'valid') return false;
      if (r.handoff_match_status && r.handoff_match_status !== 'matched') return false;
      if (r.expires_at && !(Date.parse(r.expires_at)>Date.now())) return false;
      return r.is_interacting === true;
    });
    text('interactionCount',rows.length);
    text('resultStatus',payload.errors?.length ? '部分分析暂不可用' : '分析服务已连接');
    $('results').innerHTML = rows.length ? rows.slice(0,6).map(r => `<article class="activity">${escape(r.display_name || r.person_name || r.track_id || '匿名人员')}<small>正在交互</small><b>${escape(r.instrument_name || r.instrument_id || '仪器待确认')}</b></article>`).join('') : '<p class="empty">当前暂无有效仪器交互结果</p>';
  }
  async function refresh() {
    const contract = window.__LAB_RESULT_CONTRACT__;
    const resultUrl = '/api/interaction-results/latest' + (contract ? `?contract=${encodeURIComponent(contract)}` : '');
    const [dashboard, hardware, activity] = await Promise.allSettled([get(`${api}/api/dashboard`),get(window.__LAB_HARDWARE_URL__ || `${api}/api/environment`),get(resultUrl)]);
    if (dashboard.status === 'fulfilled') {
      state(dashboard.value); text('connection','实时数据已连接'); $('connection').className='status live';
    } else {
      state({}); text('updated','数据连接中断'); text('connection','数据连接中断 · 正在重试'); $('connection').className='status error';
    }
    environment(hardware.status === 'fulfilled' ? hardware.value : dashboard.status === 'fulfilled' ? dashboard.value.environment : null);
    try { if (activity.status !== 'fulfilled') throw activity.reason; results(activity.value); }
    catch { text('interactionCount','—'); text('resultStatus','分析服务暂不可用'); $('results').innerHTML='<p class="empty">等待分析服务恢复</p>'; }
    setTimeout(refresh,2000);
  }
  refresh();
})();
