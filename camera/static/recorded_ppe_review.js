(() => {
  const get = id => document.getElementById(id);
  const video = get("ppeVideo");
  const canvas = document.createElement("canvas");
  const notice = get("ppeNotice");
  const cards = get("ppeCandidates");
  const file = get("ppeVideoFile");
  const saved = get("ppeSavedVideo");
  const cameraSelect = get("ppeVideoCamera");
  const position = get("ppeSavedPosition");
  const base = "/api/management/face-calibration/";
  let sessions = [];
  let prepared = [];
  let activeRecording;
  const buttons = [get("ppeSnap"), get("ppeAuto")];
  const labels = {worn:"已穿戴", not_worn:"未穿戴", unknown:"未知"};
  const selected = [];
  const visibleKeys = new Set();
  const bits = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4];
  let url;

  function resetResults() {
    cards.replaceChildren();
    visibleKeys.clear();
    selected.length = 0;
    allStatus.textContent = "";
  }

  const candidateKey = (candidate,camera) => camera + ":" +
    (candidate.frame ?? candidate.second.toFixed(3));

  file.addEventListener("change", () => {
    saved.value = "";
    activeRecording = null;
    get("ppeSavedPreview").hidden = true;
    video.hidden = false;
    if (url) URL.revokeObjectURL(url);
    resetResults();
    if (!file.files.length) {
      video.removeAttribute("src");
      video.load();
      return;
    }
    url = URL.createObjectURL(file.files[0]);
    video.src = url;
  });

  function savedFrameUrl(recording, index) {
    return base + "ppe-recording-frame/" +
      [recording.subject_id,recording.session_id,recording.camera].map(encodeURIComponent).join("/") +
      "?index=" + index;
  }

  function previewSaved() {
    if (!activeRecording) return;
    get("ppeSavedFrame").src = savedFrameUrl(activeRecording,Number(position.value));
    get("ppeSavedTime").textContent = (Number(position.value)/activeRecording.fps).toFixed(1)+" 秒";
  }

  saved.addEventListener("change", () => {
    activeRecording = sessions.find(row => row.key === saved.value)?.streams[0];
    if (url) { URL.revokeObjectURL(url); url = null; }
    file.value = "";
    video.removeAttribute("src");
    video.load();
    video.hidden = Boolean(activeRecording);
    get("ppeSavedPreview").hidden = !activeRecording;
    resetResults();
    if (!activeRecording) return;
    cameraSelect.value = activeRecording.camera;
    position.max = activeRecording.frames-1;
    position.value = 0;
    previewSaved();
    const extracted = prepared.find(row=>row.recording===saved.value);
    if (extracted) {
      extracted.candidates.forEach(candidate=>render(candidate,saved.value,candidate.camera));
      notice.textContent = `这段录像已截取 ${extracted.candidates.length} 张精选照片，可直接审核。`;
    }
  });
  position.addEventListener("change",previewSaved);
  cameraSelect.addEventListener("change", () => {
    if (!saved.value) return;
    const next=sessions.find(row=>row.key===saved.value).streams.find(row=>row.camera===cameraSelect.value);
    if (!next) { cameraSelect.value=activeRecording.camera; return; }
    activeRecording=next;
    position.value=0;
    position.max=next.frames-1;
    previewSaved();
  });

  Promise.all(["ppe-recordings","ppe-prepared"].map(path=>fetch(base+path,{cache:"no-store"}).then(response=>{
    if (!response.ok) throw new Error();
    return response.json();
  }))).then(([data,extracted])=>{
    prepared=extracted;
    sessions=data.sessions.map(row=>({
      key:[row.subject_id,row.session_id].join("/"),
      streams:Object.entries(row.streams).filter(([,stream])=>stream.video_file && stream.frames>0 && stream.fps>0)
        .map(([camera,stream])=>({
          subject_id:row.subject_id,session_id:row.session_id,camera,frames:stream.frames,fps:stream.fps,
          offset:(Date.parse(stream.first_frame_at)-Date.parse(row.started_at))/1000 || 0,
        })),
    })).filter(row=>row.streams.length);
    saved.replaceChildren(new Option(sessions.length?"选择已录制录像":"没有可用录像",""));
    sessions.forEach(row=>saved.add(new Option(row.key,row.key)));
    const requestedRecording = new URLSearchParams(location.search).get("ppeRecording");
    const initial = sessions.find(row => row.key === requestedRecording) ||
      sessions.find(row=>prepared.some(batch=>batch.recording===row.key && batch.candidates.length));
    if (initial && !file.files.length && !saved.value) {
      saved.value = initial.key;
      saved.dispatchEvent(new Event("change"));
    }
  }).catch(()=>saved.replaceChildren(new Option("录像列表暂时无法读取","")));

  async function recordedSnapshot(recording,index) {
    const response=await fetch(savedFrameUrl(recording,index));
    if (!response.ok) throw new Error("录像帧无法读取，帧编号 "+index);
    const bitmap=await createImageBitmap(await response.blob());
    try {
      const scale=Math.min(1,1920/bitmap.width);
      canvas.width=Math.round(bitmap.width*scale);
      canvas.height=Math.round(bitmap.height*scale);
      canvas.getContext("2d").drawImage(bitmap,0,0,canvas.width,canvas.height);
      return {second:index/recording.fps,jpeg:canvas.toDataURL("image/jpeg",.88).split(",")[1]};
    } finally {
      bitmap.close();
    }
  }

  async function request(action, body) {
    const response = await fetch("/api/management/face-calibration/ppe-" + action, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "请求失败");
    return data;
  }

  function snapshot() {
    const scale = Math.min(1, 1920/video.videoWidth);
    canvas.width = Math.round(video.videoWidth*scale);
    canvas.height = Math.round(video.videoHeight*scale);
    canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);
    return {second:video.currentTime,jpeg:canvas.toDataURL("image/jpeg",.88).split(",")[1]};
  }

  function seek(second) {
    if (Math.abs(video.currentTime-second)<.01) return Promise.resolve();
    return new Promise((resolve,reject) => {
      const finish = () => { clearTimeout(timer); resolve(); };
      const timer = setTimeout(() => {
        video.removeEventListener("seeked", finish);
        reject(new Error("录像定位超时"));
      }, 5000);
      video.addEventListener("seeked",finish,{once:true});
      video.currentTime=second;
    });
  }

  function render(candidate, recording, camera) {
    const key = candidateKey(candidate,camera);
    if (visibleKeys.has(key)) return;
    visibleKeys.add(key);
    const card = document.createElement("article");
    card.className = "ppe-card";
    const title = document.createElement("p");
    title.className = "ppe-card-meta";
    title.textContent = `${camera} · ${candidate.second.toFixed(1)} 秒`;
    const hints = document.createElement("p");
    hints.className = "ppe-card-hints";
    hints.textContent = `口罩建议 ${labels[candidate.mask]} · 实验服建议 ${labels[candidate.lab_coat]}（${Math.round(candidate.coat_score*100)}%）`;
    if (candidate.matches?.length) {
      const matched = candidate.matches.map(key => key === "mask" ? "口罩" : "实验服");
      hints.textContent += ` · 清晰候选：${matched.join("、")}`;
    }
    if (candidate.similarity >= .9) hints.textContent += ` · 相似度 ${Math.round(candidate.similarity*100)}%`;
    card.append(title);
    const photos = document.createElement("div");
    photos.className = "ppe-card-photos";
    for (const [key,alt] of [["person","人员裁图"],["head","头部裁图"]]) {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.alt = alt;
      image.loading = "lazy";
      image.src = "data:image/jpeg;base64,"+candidate[key];
      const caption = document.createElement("figcaption");
      caption.textContent = alt;
      figure.append(image,caption);
      photos.append(figure);
    }
    card.append(photos,hints);
    const controls = document.createElement("div");
    controls.className = "ppe-card-controls";
    const fields = {};
    for (const [key,label] of [["mask","口罩"],["lab_coat","实验服"]]) {
      const field = document.createElement("label");
      field.textContent = label;
      const select = document.createElement("select");
      select.add(new Option("暂不标注",""));
      for (const [value,name] of Object.entries(labels)) select.add(new Option(name,value));
      fields[key] = select;
      field.append(select);
      controls.append(field);
    }
    const save = document.createElement("button");
    save.textContent = "确认入库";
    save.addEventListener("click", async () => {
      const confirmed = Object.fromEntries(Object.entries(fields).filter(([,s])=>s.value).map(([k,s])=>[k,s.value]));
      if (!Object.keys(confirmed).length) { notice.textContent="请至少选择一个人工标签。"; return; }
      save.disabled=true;
      try {
        let source = candidate.source;
        if (!source && candidate.frame !== undefined) {
          const query = new URLSearchParams({recording,source:camera+"-"+candidate.frame});
          const response = await fetch(base+"ppe-prepared-all?"+query);
          if (!response.ok) throw new Error("候选原图无法读取");
          ({source} = await response.json());
        }
        await request("save",{recording,camera,second:candidate.second,source,bbox:candidate.bbox,labels:confirmed});
        save.textContent="已入库";
      } catch (error) {
        notice.textContent=error.message;
        save.disabled=false;
      }
    });
    controls.append(save);
    card.append(controls);
    cards.append(card);
  }

  function duplicate(candidate) {
    const group = candidate.camera + ":" + candidate.matches.join(",");
    return selected.some(prior => {
      if (prior.group !== group) return false;
      let distance = 0;
      for (let i=0; i<candidate.fingerprint.length; i++) {
        distance += bits[parseInt(candidate.fingerprint[i],16) ^ parseInt(prior.hash[i],16)];
      }
      return distance <= 12;
    });
  }

  async function collect(auto) {
    const stored=activeRecording;
    const camera=stored?.camera || cameraSelect.value;
    if ((!stored && video.readyState<2) || !camera) { notice.textContent="请选择录像和原摄像头。"; return; }
    const recording=stored ? saved.value : file.files[0].name;
    saved.disabled=true;
    file.disabled=true;
    buttons.forEach(button=>button.disabled=true);
    video.pause();
    const start=video.currentTime;
    try {
      notice.textContent="正在截取并生成候选…";
      let total=0;
      if (auto) {
        const streams=stored ? sessions.find(row=>row.key===recording).streams : [];
        const duration=stored ? Math.max(...streams.map(row=>row.offset+row.frames/row.fps)) : video.duration;
        for(let second=0;second<duration;){
          const frames=[], cameras=[];
          for(let tick=0;tick<(stored?4:12) && second<duration;tick++,second++){
            if (stored) {
              for (const stream of streams) {
                const index=Math.round((second-stream.offset)*stream.fps);
                if (index<0 || index>=stream.frames) continue;
                const frame=await recordedSnapshot(stream,index);
                cameras.push({second:frame.second+stream.offset,camera:stream.camera,frame:index});
                frames.push({...frame,second:frames.length});
              }
            }
            else {
              await seek(second);
              frames.push(snapshot());
            }
          }
          if (!frames.length) continue;
          notice.textContent=`正在扫描 ${Math.min(Math.round(second/duration*100),100)}% · 已保留 ${total} 张`;
          const {candidates}=await request("scan",{frames,automatic:true});
          for (const candidate of candidates) {
            if (stored) Object.assign(candidate,cameras[candidate.second]);
            else candidate.camera=camera;
            if (duplicate(candidate)) continue;
            selected.push({group:candidate.camera+":"+candidate.matches.join(","),hash:candidate.fingerprint});
            render(candidate,recording,candidate.camera);
            total++;
          }
        }
        if (!stored) await seek(start);
      } else {
        const frame=stored ? await recordedSnapshot(stored,Number(position.value)) : snapshot();
        const {candidates}=await request("scan",{frames:[frame]});
        if (stored) candidates.forEach(candidate=>candidate.frame=Number(position.value));
        candidates.forEach(candidate=>render(candidate,recording,camera));
        total=candidates.length;
      }
      notice.textContent=total ? `生成 ${total} 张候选，请检查并确认标签。` :
        (auto ? "整段已扫描，本次没有新增合格候选。" : "未找到可用的单人人员框。");
    } catch(error) {
      notice.textContent=error.message;
    } finally {
      file.disabled=false;
      saved.disabled=false;
      buttons.forEach(button=>button.disabled=false);
    }
  }

  buttons[0].addEventListener("click",()=>collect(false));
  buttons[1].addEventListener("click",()=>collect(true));
})();
