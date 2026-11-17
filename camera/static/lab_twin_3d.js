/* A WebGL view of the floor map. Coordinates are shared with the calibration view. */
const LabTwin3D = (() => {
  let view;
  let unavailable = false;

  function create(host) {
    const T = window.THREE;
    const renderer = new T.WebGLRenderer({antialias:true, alpha:true, preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25));
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFShadowMap;
    const root = document.createElement("div");
    root.className = "twin-webgl";
    const labels = document.createElement("div");
    labels.className = "twin-webgl-labels";
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "tool-link twin-reset";
    reset.textContent = "重置视角";
    const caption = document.createElement("div");
    caption.className = "twin-stage-caption";
    caption.textContent = "实验室微缩场景";
    const hint = document.createElement("div");
    hint.className = "twin-stage-hint";
    hint.textContent = "拖动旋转 · 滚轮缩放";
    root.append(renderer.domElement, labels, caption, hint, reset);
    host.append(root);
    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(38, 1, .1, 140);
    const content = new T.Group();
    scene.add(content, new T.HemisphereLight(0xf1f8ff, 0x607486, 2.25));
    const lines = new T.Group();
    scene.add(lines);
    const lineStyle = {
      grid: new T.LineBasicMaterial({color:0x607a8a, transparent:true, opacity:.32}),
      edge: new T.LineBasicMaterial({color:0x53c9ff, transparent:true, opacity:.86}),
      coverage: new T.LineDashedMaterial({color:0x55b6ed, transparent:true, opacity:.38,
        dashSize:.16, gapSize:.12}),
      trail: new T.LineDashedMaterial({color:0x79ebff, transparent:true, opacity:.9,
        dashSize:.16, gapSize:.09}),
      person: new T.LineBasicMaterial({color:0x61d9ff, transparent:true, opacity:.8}),
      visitor: new T.LineBasicMaterial({color:0xffb57e, transparent:true, opacity:.8}),
    };
    function drawLines(vertices, style) {
      if (!vertices.length) return;
      const buffer = new T.BufferGeometry();
      buffer.setAttribute("position", new T.Float32BufferAttribute(vertices,3));
      const object = new T.LineSegments(buffer,lineStyle[style]);
      object.computeLineDistances();
      lines.add(object);
    }
    const key = new T.DirectionalLight(0xfff8ee, 2.8);
    key.position.set(5, 12, 4);
    key.castShadow = true;
    Object.assign(key.shadow.camera, {left:-11,right:11,top:11,bottom:-11,near:.5,far:35});
    key.shadow.mapSize.set(512,512);
    key.shadow.normalBias = .025;
    scene.add(key);
    const rim = new T.DirectionalLight(0xb8ddff, 1.25);
    rim.position.set(-6, 5, -7);
    scene.add(rim);
    const blueEdge = new T.MeshBasicMaterial({color:0x70dcff, toneMapped:false});
    const materials = new Map();
    const geometry = new Map();
    const tags = [];
    const overview = {yaw:.57, pitch:.82, zoom:1};
    let {yaw,pitch,zoom} = overview;
    let extent = 12;
    const target = new T.Vector3(0,.35,0);
    let drag;
    let contextLost = false;
    let width = 0, height = 0, signature = "";

    function draw() {
      if (contextLost || root.hidden || !root.clientWidth || !root.clientHeight) return;
      const w = root.clientWidth, h = root.clientHeight;
      if (w !== width || h !== height) {
        renderer.setSize(w,h,false);
        width = w;
        height = h;
      }
      camera.aspect = w/h;
      const distance = extent * 1.45 * zoom / Math.min(1,camera.aspect);
      camera.position.set(distance*Math.cos(pitch)*Math.sin(yaw), distance*Math.sin(pitch),
        distance*Math.cos(pitch)*Math.cos(yaw)).add(target);
      camera.lookAt(target);
      camera.updateProjectionMatrix();
      renderer.render(scene,camera);
      for (const {position, node} of tags) {
        const screen = position.clone().project(camera);
        node.hidden = screen.z < -1 || screen.z > 1 || Math.abs(screen.x) > .98 || Math.abs(screen.y) > .92;
        node.style.left = `${(screen.x+1)*50}%`;
        node.style.top = `${(1-screen.y)*50}%`;
      }
    }
    reset.onclick = () => { ({yaw,pitch,zoom}=overview); target.set(0,.35,0); draw(); };
    const canvas = renderer.domElement;
    canvas.addEventListener("webglcontextlost", event => {
      event.preventDefault();
      contextLost = true;
      root.hidden = true;
    });
    canvas.addEventListener("webglcontextrestored", () => {
      contextLost = false;
      width = height = 0;
    });
    canvas.setAttribute("aria-label","三维实验室，拖动旋转，滚轮缩放");
    canvas.addEventListener("pointerdown", event => {
      if (!event.isPrimary || event.button !== 0) return;
      drag = [event.clientX,event.clientY];
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      if (!drag || !event.isPrimary) return;
      yaw -= (event.clientX-drag[0])*.009;
      pitch = Math.max(.18,Math.min(1.48,pitch+(event.clientY-drag[1])*.009));
      drag = [event.clientX,event.clientY];
      draw();
    });
    canvas.addEventListener("lostpointercapture", () => { drag = null; });
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      zoom = Math.max(.16,Math.min(2.2,zoom*Math.exp(event.deltaY*.001)));
      draw();
    }, {passive:false});
    new ResizeObserver(draw).observe(root);

    function mesh(parent, shape, size, color, x,y,z, opacity=1) {
      const key = `${shape}:${size.join(",")}`;
      if (!geometry.has(key)) {
        const constructors = {box:T.BoxGeometry,cylinder:T.CylinderGeometry,
          sphere:T.SphereGeometry,capsule:T.CapsuleGeometry,ring:T.RingGeometry};
        if (shape === "housing") {
          const [w,h,d,r] = size;
          const a=w/2-r, b=h/2-r;
          const profile = new T.Shape();
          profile.moveTo(-a,-b); profile.lineTo(a,-b);
          profile.lineTo(a,b); profile.lineTo(-a,b); profile.closePath();
          const solid = new T.ExtrudeGeometry(profile,{
            depth:d-2*r, bevelEnabled:true, bevelSize:r, bevelThickness:r,
            bevelSegments:3, steps:1,
          });
          solid.translate(0,0,-(d-2*r)/2);
          geometry.set(key,solid);
        } else {
          geometry.set(key,new constructors[shape](...size));
        }
      }
      const surface = `${color}:${opacity}`;
      if (!materials.has(surface)) materials.set(surface,new T.MeshStandardMaterial({
        color,roughness:.52,metalness:.12,opacity,transparent:opacity<1,depthWrite:opacity===1,
      }));
      const result = new T.Mesh(geometry.get(key),materials.get(surface));
      result.position.set(x,y,z);
      result.castShadow = opacity===1;
      result.receiveShadow = true;
      parent.add(result);
      return result;
    }
    const box = (parent,w,h,d,color,x,y,z,opacity) => mesh(parent,"box",[w,h,d],color,x,y,z,opacity);
    const cylinder = (parent,r,h,color,x,y,z) => mesh(parent,"cylinder",[r,r,h,28],color,x,y,z);
    function lightStrip(w,h,d,x,y,z) {
      const strip = box(content,w,h,d,0x70dcff,x,y,z);
      strip.material = blueEdge;
      strip.castShadow = false;
      strip.receiveShadow = false;
    }

    function miniatureBase(width, depth) {
      const x = width/2, z = depth/2;
      box(content,width+.42,.18,depth+.42,0x162e43,0,-.39,0);
      box(content,width+.23,.12,depth+.23,0x486a80,0,-.27,0);
      box(content,width,.24,depth,0xa7b5bd,0,-.12,0);
      for (const side of [-1,1]) {
        lightStrip(width+.24,.019,.026,0,-.237,side*(z+.112));
        lightStrip(.026,.019,depth+.24,side*(x+.112),-.237,0);
        for (const end of [-1,1]) {
          box(content,.39,.055,.39,0x497b99,side*(x-.17),.005,end*(z-.17));
        }
      }
    }
    function cabinetFront(parent, width, height, x, z) {
      const bays = Math.max(1, Math.round(width / .75));
      const bay = width / bays;
      for (let i=0; i<bays; i++) {
        const panelX = x-width/2+(i+.5)*bay;
        box(parent,bay-.035,height,.035,i%3===2 ? 0xa2bed2 : 0xd7e4ec,panelX,.12+height/2,z);
        box(parent,.17,.027,.03,0x263f54,panelX,.03+height,z+.029);
      }
    }

    function glassBoundary(width, depth) {
      const bays = Math.ceil(width/1.7);
      const bay = width/bays;
      const z = -depth/2;
      box(content,width,.61,.16,0xd2e2ec,0,.305,z);
      box(content,width,.055,.07,0xc7dfef,0,2.03,z);
      for (let i=0; i<=bays; i++) {
        const x = -width/2+i*bay;
        box(content,.052,1.46,.055,0xabc6d8,x,1.31,z);
        if (i<bays) box(content,bay-.055,1.39,.025,0x9fd4ea,x+bay/2,1.31,z,.14);
      }
      const sideX = -width/2;
      const sideBays = Math.ceil(depth/1.7);
      const sideBay = depth/sideBays;
      box(content,.16,.61,depth,0xd2e2ec,sideX,.305,0);
      box(content,.056,.07,depth,0xc7dfef,sideX,2.03,0);
      for (let i=0; i<sideBays; i++) {
        box(content,.028,1.39,sideBay-.055,0x9fd4ea,sideX,1.31,z+(i+.5)*sideBay,.14);
        box(content,.055,1.46,.055,0xabc6d8,sideX,1.31,z+(i+1)*sideBay);
      }
    }

    const isTallCabinet = kind => kind === "reagent_cabinet" || kind === "chemical_cabinet";
    const isFloorEquipment = kind => kind === "fume_hood" || isTallCabinet(kind);
    const tallCabinetWidth = 1.16;
    const tallCabinetDepth = .62;
    const wallCabinetCenter = .10 + tallCabinetDepth/2;
    const hoodWidth = 1.24 * .94;

    function upperStorage(fixture,x,z,top,occupied) {
      const count = Math.floor(fixture.width/.92);
      for (let i=0; i<count; i++) {
        const bx = x+(i-(count-1)/2)*.92;
        if (occupied.some(center => Math.abs(bx-center) < (tallCabinetWidth+.895)/2)) continue;
        const bz = z-fixture.height*.25;
        box(content,.895,.59,.36,0xbeced9,bx,top+.65,bz);
        box(content,.80,.46,.023,i%3===1 ? 0x779cae : 0xd5e2ea,bx,top+.65,bz+.19);
        box(content,.033,.13,.026,0x385b72,bx+.31,top+.62,bz+.21);
      }
      lightStrip(fixture.width-.30,.018,.025,x,top+.348,z-fixture.height*.25+.18);
    }

    const capsule = (parent,r,length,color,x,y,z) =>
      mesh(parent,"capsule",[r,length,5,12],color,x,y,z);

    function personModel(x, z, identified, heading) {
      const body = new T.Group();
      body.position.set(x,0,z);
      body.rotation.y = heading;
      content.add(body);
      const marker = mesh(body,"ring",[.29,.33,40],identified ? 0x62dfbe : 0xffaf73,0,.018,0);
      marker.rotation.x = -Math.PI/2;
      marker.castShadow = false;
      const trousers = 0x2d4158;
      const coat = 0xf2f5f7;
      const skin = 0xd9aa8c;
      for (const side of [-1,1]) {
        capsule(body,.078,.45,trousers,side*.105,.36,0);
        capsule(body,.067,.15,0x24354a,side*.105,.07,.045).rotation.x = Math.PI/2;
        capsule(body,.066,.38,coat,side*.235,.96,.01).rotation.z = side*.14;
        mesh(body,"sphere",[.052,14,10],skin,side*.268,.72,.013);
      }
      const coatBody = mesh(body,"cylinder",[.207,.184,.65,20],coat,0,.89,0);
      coatBody.scale.z = .66;
      const shoulders = mesh(body,"sphere",[1,20,14],coat,0,1.175,0);
      shoulders.scale.set(.209,.112,.134);
      // The avatar uses a standard lab coat as a visual convention.
      box(body,.105,.185,.015,0x263c51,0,1.145,.129);
      for (const side of [-1,1]) {
        box(body,.047,.218,.024,0xffffff,side*.055,1.132,.145).rotation.z = side*.33;
        box(body,.088,.069,.016,0xd4e2eb,side*.102,.77,.128);
      }
      box(body,.012,.41,.014,0xbccdd8,0,.833,.13);
      for (const y of [.79,.91,1.03]) {
        mesh(body,"sphere",[.013,8,6],0xfafcff,.017,y,.145);
      }
      capsule(body,.055,.045,skin,0,1.29,0);
      const head = mesh(body,"sphere",[1,20,14],skin,0,1.43,0);
      head.scale.set(.105,.14,.10);
      const hair = mesh(body,"sphere",[1,20,14],0x293748,0,1.49,-.022);
      hair.scale.set(.109,.09,.099);
      box(body,.067,.038,.018,0xc6dbe9,.112,1.061,.134);
    }

    function trackingOutline(x, z, identified) {
      const vertices = [];
      for (const sideX of [-1,1]) for (const sideZ of [-1,1]) {
        const px = x+sideX*.34, pz = z+sideZ*.29;
        for (const y of [.035,1.65]) {
          vertices.push(px,y,pz,px-sideX*.12,y,pz,px,y,pz,px,y,pz-sideZ*.12,
            px,y,pz,px,y+(y<.1 ? .25 : -.25),pz);
        }
      }
      drawLines(vertices,identified ? "person" : "visitor");
    }

    function instrumentModel(kind, x, y, z) {
      const model = new T.Group();
      model.position.set(x,y,z);
      content.add(model);
      const part = (w,h,d,color,px,py,pz,opacity) => box(model,w,h,d,color,px,py,pz,opacity);
      const housing = (w,h,d,r,color,px,py,pz) =>
        mesh(model,"housing",[w,h,d,r],color,px,py,pz);
      const steel = 0xadbcc5, graphite = 0x263644, display = 0x183e60;
      const white = 0xe0e7eb, blue = 0x28648f;
      const screen = (px, py, pz, width=.16) => {
        part(width,.055,.019,graphite,px,py,pz);
        part(width*.78,.033,.012,0x8ed8e8,px,py,pz+.013);
      };
      const knob = (px, py, pz, color=graphite, radius=.025) =>
        cylinder(model,radius,.024,color,px,py,pz).rotation.x = Math.PI/2;
      if (kind === "balance") {
        housing(.56,.115,.55,.024,0xe2e0ce,0,.078,0);
        part(.50,.025,.48,graphite,0,.027,0);
        const console = part(.46,.052,.16,0x32658b,0,.098,.242);
        console.rotation.x = -.24;
        part(.18,.012,.064,display,-.071,.127,.245).rotation.x = -.24;
        for (const px of [.067,.113,.159]) part(.025,.014,.032,0xc5d7dd,px,.123,.253);
        cylinder(model,.112,.035,steel,0,.162,-.049);
        for (const px of [-.202,.202]) {
          part(.012,.33,.38,0x8ddbf3,px,.303,-.046,.19);
          for (const pz of [-.238,.145]) part(.025,.35,.025,0xd8dee1,px,.31,pz);
        }
        for (const pz of [-.238,.145]) part(.398,.32,.012,0x8ddbf3,0,.303,pz,.18);
        housing(.43,.028,.412,.009,0xd8dee1,0,.488,-.046);
        part(.015,.14,.024,0xb8d5df,.12,.31,.16);
        part(.37,.022,.024,0xa1b5be,0,.164,.15);
      } else if (kind === "cleaner") {
        housing(.58,.31,.46,.024,steel,0,.17,0);
        part(.54,.032,.42,graphite,0,.31,0);
        for (const px of [-.27,.27]) part(.039,.09,.46,steel,px,.359,0);
        for (const pz of [-.21,.21]) part(.52,.09,.035,steel,0,.359,pz);
        for (let i=-4; i<=4; i++) part(.009,.012,.37,0x80939d,i*.05,.345,0);
        for (let i=-3; i<=3; i++) part(.46,.012,.009,0xa6b8c0,0,.348,i*.049);
        part(.19,.053,.019,graphite,0,.23,.239);
        part(.16,.088,.017,display,.19,.115,.24);
        for (const px of [.146,.191,.235]) {
          part(.025,.018,.012,0x91bed4,px,.106,.254);
        }
        part(.45,.021,.025,0xd8e5ea,0,.403,.21);
      } else if (kind === "centrifuge") {
        housing(.63,.225,.53,.033,0xdedccc,0,.129,0);
        part(.58,.046,.48,graphite,0,.034,0);
        housing(.61,.089,.51,.027,0xe7e5d5,0,.271,-.018);
        cylinder(model,.028,.012,graphite,0,.322,-.05);
        const panel = part(.48,.18,.035,0xbfc1b3,0,.156,.268);
        panel.rotation.x = -.38;
        part(.275,.069,.021,display,-.035,.192,.289).rotation.x = -.38;
        for (let i=0; i<5; i++) {
          mesh(model,"sphere",[.019,10,6],i===4 ? 0xd98555 : 0x506c89,
            -.13+i*.065,.115,.308);
        }
        for (let i=0; i<6; i++) part(.017,.055,.014,0x71818c,-.22+i*.032,.083,.266);
      } else if (kind === "heater" || kind === "stirrer" || kind === "single_stirrer" || kind === "scale") {
        const wide = kind === "stirrer";
        const width = wide ? .78 : .49;
        const depth = wide ? .57 : .45;
        const body = kind === "heater" || kind === "scale" ? white : blue;
        housing(width,.14,depth,.022,body,0,.083,0);
        if (kind === "single_stirrer") {
          cylinder(model,.16,.035,steel,0,.177,-.048);
        } else {
          housing(width-.045,.035,depth-.13,.01,kind === "stirrer" ? white : steel,0,.177,-.049);
        }
        if (wide) {
          const panel = part(width-.03,.115,.033,blue,0,.10,.275);
          panel.rotation.x = -.22;
          for (const row of [0,1]) {
            for (let i=0; i<5; i++) {
              const px = .023+i*.062;
              knob(px,.072+row*.049,.302,steel,.017);
              part(.042,.019,.012,display,px-.34,.072+row*.049,.305);
            }
          }
        } else {
          screen(-.073,.102,depth/2);
          knob(.103,.102,depth/2+.021,kind === "heater" ? 0xc74639 : graphite);
        }
      } else if (kind === "fume_hood") {
        model.scale.set(.94,1.29,1.24);
        model.rotation.y = Math.PI;
        part(1.24,.70,.56,white,0,.35,0);
        for (const side of [-1,1]) {
          part(.57,.59,.025,steel,side*.30,.35,.285);
          part(.028,.11,.035,graphite,side*.07,.39,.307);
        }
        part(1.24,.83,.055,steel,0,1.12,-.26);
        for (const side of [-1,1]) {
          part(.065,.86,.56,white,side*.59,1.14,0);
          part(.026,.72,.035,steel,side*.53,1.16,.264);
        }
        part(1.24,.14,.57,white,0,1.565,0);
        part(1.13,.043,.53,graphite,0,.744,0);
        part(1.05,.43,.014,0x8edbf2,0,1.235,.263,.27);
        part(1.07,.026,.03,steel,0,1.017,.267);
        screen(.42,1.57,.292,.16);
      } else if (kind === "freeze_dryer") {
        housing(.61,.38,.49,.034,white,0,.203,0);
        part(.19,.33,.032,graphite,.204,.208,.25).rotation.x = -.22;
        screen(.21,.30,.273,.13);
        cylinder(model,.16,.065,graphite,0,.408,0);
        mesh(model,"cylinder",[.15,.15,.45,28],0x97d8e8,0,.657,0,.28);
        for (const py of [.475,.58,.685,.79]) cylinder(model,.133,.019,steel,0,py,0);
        cylinder(model,.151,.026,steel,0,.884,0);
      } else if (kind === "sonicator") {
        housing(.44,.57,.45,.02,white,0,.303,0);
        part(.39,.12,.44,graphite,0,.534,0);
        part(.29,.31,.025,graphite,0,.237,.229);
        part(.26,.29,.017,0x728a9a,0,.237,.248,.48);
        cylinder(model,.025,.145,steel,0,.28,.263);
        cylinder(model,.075,.074,steel,0,.634,0);
        screen(.045,.531,.233,.13);
      } else if (kind === "oven" || kind === "vacuum_oven" || kind === "dryer" ||
                 kind === "reagent_cabinet" || kind === "chemical_cabinet") {
        const storage = isTallCabinet(kind);
        const height = storage ? 1.9 : .62;
        const width = storage ? tallCabinetWidth : .53;
        const depth = storage ? tallCabinetDepth : .46;
        const front = depth/2;
        const color = kind === "chemical_cabinet" ? 0xd8aa3e : kind === "oven" ? blue : white;
        housing(width,height,depth,.02,color,0,height/2,0);
        const doors = storage ? [-width/4,width/4] : [-.045];
        for (const px of doors) {
          const panelWidth = storage ? width/2-.022 : .36;
          part(panelWidth,height-.11,.027,color,px,height/2,front+.01);
          if (kind !== "chemical_cabinet") {
            part(panelWidth*.62,height*.52,.022,0x274653,px,height*.56,front+.028);
            for (const fraction of [.43,.56,.69]) part(panelWidth*.57,.012,.018,steel,px,height*fraction,front+.041);
          }
          part(.024,storage ? .21 : .115,.031,steel,px+panelWidth*.37,height*.47,front+.048);
        }
        if (!storage) screen(.19,height-.105,.248,.09);
      } else if (kind === "pump" || kind === "water_system" || kind === "bath") {
        const tall = kind === "water_system";
        const height = tall ? .58 : .28;
        housing(.51,height,.4,.026,tall ? white : kind === "pump" ? blue : steel,0,height/2+.015,0);
        if (tall) {
          screen(.09,.47,.208,.22);
          part(.11,.41,.12,white,-.31,.216,.045);
          part(.12,.026,.12,blue,-.31,.42,.13);
        } else if (kind === "pump") {
          housing(.47,.124,.36,.02,white,0,.335,0);
          for (const px of [-.08,.08]) knob(px,.337,.2,steel,.043);
          part(.11,.038,.1,graphite,0,.411,0);
        } else {
          part(.46,.034,.36,steel,0,.31,0);
          part(.13,.038,.03,graphite,0,.345,0);
          screen(.12,.16,.21,.15);
        }
      } else if (kind === "viscometer" || kind === "overhead_stirrer") {
        housing(.41,.06,.4,.02,graphite,0,.04,0);
        cylinder(model,.025,.63,steel,-.125,.365,-.1);
        part(.21,.052,.11,steel,-.033,.48,-.08);
        housing(.18,.2,.17,.017,kind === "viscometer" ? white : blue,.065,.51,.006);
        screen(.065,.537,.094,.105);
        cylinder(model,.014,.25,steel,.065,.285,.025);
        mesh(model,"cylinder",[.088,.07,.135,24],0x9ed7e1,.065,.14,.025,.37);
      } else if (kind === "computer") {
        model.rotation.y = Math.PI;
        part(.42,.25,.032,graphite,0,.25,-.07);
        part(.39,.22,.012,display,0,.255,-.048);
        part(.045,.12,.044,steel,0,.085,-.07);
        part(.18,.023,.14,graphite,0,.015,-.05);
        part(.31,.019,.115,graphite,-.027,.016,.135);
        part(.045,.022,.073,steel,.183,.017,.135);
      } else if (kind === "grinder") {
        housing(.36,.26,.36,.025,blue,0,.145,0);
        cylinder(model,.139,.22,steel,0,.38,0);
        cylinder(model,.155,.038,graphite,0,.505,0);
        knob(.064,.15,.2,white);
      } else {
        housing(.46,.24,.38,.025,white,0,.14,0);
        screen(0,.165,.197);
      }
      return new T.Box3().setFromObject(model).max.y;
    }

    function tag(name, x,y,z, kind, id) {
      const node = document.createElement(kind === "equipment" ? "button" : "span");
      node.className = `twin-tag ${kind}`;
      node.textContent = name;
      if (kind === "equipment") {
        node.type = "button";
        node.title = `查看${name}三维模型`;
        node.onclick = () => {
          target.set(x,y-.43,z);
          pitch=.48;
          yaw=.38;
          zoom=.17;
          draw();
        };
      }
      labels.append(node);
      tags.push({node,position:new T.Vector3(x,y,z),id});
    }

    function update(map, equipment, personLabel, trails) {
      const wasHidden = root.hidden;
      root.hidden = false;
      const people = (map.people || []).filter(person =>
        person.map_visible !== false && [person.x,person.y].every(Number.isFinite));
      const next = JSON.stringify([map.width_m,map.height_m,map.fixtures,map.cameras,equipment,
        people.map(person => [person.x,person.y,person.identified,personLabel(person),
          trails.get(person.track_id)])]);
      if (next === signature) {
        if (wasHidden) draw();
        return;
      }
      signature = next;
      const centerX = map.width_m/2, centerZ = map.height_m/2;
      extent = Math.max(map.width_m,map.height_m);
      const heights = new Map();
      content.clear();
      for (const object of lines.children) object.geometry.dispose();
      lines.clear();
      labels.replaceChildren();
      tags.length = 0;
      miniatureBase(map.width_m,map.height_m);
      glassBoundary(map.width_m,map.height_m);
      const grid = [];
      for (let x = 1; x < map.width_m; x++) grid.push(x-centerX,.008,-centerZ,x-centerX,.008,centerZ);
      for (let z = 1; z < map.height_m; z++) grid.push(-centerX,.008,z-centerZ,centerX,.008,z-centerZ);
      drawLines(grid,"grid");
      drawLines([-centerX,.02,-centerZ,centerX,.02,-centerZ,
        centerX,.02,-centerZ,centerX,.02,centerZ,centerX,.02,centerZ,-centerX,.02,centerZ,
        -centerX,.02,centerZ,-centerX,.02,-centerZ],"edge");
      const coverage = [];
      for (const [id,camera] of Object.entries(map.cameras || {})) {
        if (!camera.position?.every(Number.isFinite)) continue;
        const [mx,mz] = camera.position;
        const x = mx-centerX, z = mz-centerZ;
        box(content,.20,1.58,.20,0x294960,x,.79,z);
        box(content,.38,.20,.28,0xa1c7dc,x,1.65,z);
        box(content,.15,.095,.06,0x176696,x,1.65,z+.16);
        lightStrip(.036,.025,.013,x+.112,1.65,z+.148);
        tag(camera.label || id,x,1.94,z,"camera");
        const fov = camera.fov || [];
        for (let i=0; i<fov.length; i++) {
          const a=fov[i], b=fov[(i+1)%fov.length];
          if (![...a,...b].every(Number.isFinite)) continue;
          coverage.push(a[0]-centerX,.035,a[1]-centerZ,b[0]-centerX,.035,b[1]-centerZ);
        }
      }
      drawLines(coverage,"coverage");
      for (const fixture of map.fixtures || []) {
        let f = fixture;
        const placed = equipment.filter(item => item.fixture_id === f.id &&
          [item.x_m,item.y_m].every(Number.isFinite));
        if (f.type !== "bench" && placed.some(item => isFloorEquipment(item.kind))) {
          const tabletop = placed.filter(item => !isFloorEquipment(item.kind));
          heights.set(f.id,f.type==="cabinet" ? 1.05 : .9);
          if (!tabletop.length) continue;
          const firstTabletop = Math.min(...tabletop.map(item => item.x_m));
          const lastTabletop = Math.max(...tabletop.map(item => item.x_m));
          const leftHood = placed.filter(item => item.kind === "fume_hood" && item.x_m < firstTabletop);
          const rightCabinets = placed.filter(item => isTallCabinet(item.kind) && item.x_m > lastTabletop);
          const start = Math.max(f.x, leftHood.length
            ? Math.max(...leftHood.map(item => item.x_m)) + hoodWidth/2
            : firstTabletop - .56);
          const end = Math.min(f.x+f.width,
            rightCabinets.length ? Math.min(...rightCabinets.map(item => item.x_m - .83)) : lastTabletop+.56);
          if (end-start <= .08) continue;
          f = {...f,x:start,width:end-start};
        }
        const x=f.x+f.width/2-centerX, z=f.y+f.height/2-centerZ;
        if (f.type === "door") {
          box(content,f.width,.024,f.height,0x70a8c3,x,.016,z);
          for (const side of [-1,1]) {
            box(content,.065,1.96,.065,0xc0d3df,x,.98,z+side*(f.height/2-.03));
          }
          box(content,.065,.065,f.height,0xc0d3df,x,1.96,z);
          continue;
        }
        const top=f.type==="cabinet" ? 1.05 : .9;
        heights.set(f.id,top);
        if (f.type==="bench") {
          for (const dx of [-f.width/2+.13,0,f.width/2-.13]) {
            for (const dz of [-f.height/2+.13,f.height/2-.13]) {
              box(content,.10,top-.08,.10,0x607887,x+dx,(top-.08)/2,z+dz);
            }
          }
          const cabinetWidth = f.width*.42;
          box(content,cabinetWidth,top-.18,f.height-.23,0xb7c9d5,x,(top-.18)/2+.06,z);
          cabinetFront(content,cabinetWidth,top-.25,x,z+(f.height-.23)/2);
          cabinetFront(content,cabinetWidth,top-.25,x,z-(f.height-.23)/2);
          // Schematic rack and glassware, placed at the end away from instrument anchors.
          const rackX = x-f.width/2+.88;
          const rackZ = z-.12;
          for (const dx of [-.59,.59]) box(content,.035,.54,.045,0x89acbc,rackX+dx,top+.27,rackZ);
          box(content,1.23,.035,.36,0x658fac,rackX,top+.48,rackZ);
          for (const dx of [-.41,-.14,.14,.41]) {
            const vial = mesh(content,"cylinder",[.075,.075,.18,12],0x78ceeb,rackX+dx,top+.59,rackZ,.54);
            vial.castShadow = false;
            cylinder(content,.041,.035,0xa0def1,rackX+dx,top+.693,rackZ);
          }
        } else {
          box(content,f.width-.08,top-.09,f.height-.08,0xa9bfcd,x,(top-.09)/2,z);
          cabinetFront(content,f.width-.08,top-.22,x,z+(f.height-.08)/2);
        }
        box(content,f.width,.09,f.height,0x405565,x,top-.045,z);
        if (f.type==="cabinet") {
          const tallCabinets = equipment.filter(item => isTallCabinet(item.kind) &&
            item.fixture_id === f.id && [item.x_m, item.y_m].every(Number.isFinite));
          upperStorage(f,x,z,top,tallCabinets.map(item => item.x_m-centerX));
        }
        box(content,f.width-.04,.013,.015,0x79b9d7,x,top-.03,z+f.height/2+.006);
        if (f.type==="bench") {
          lightStrip(f.width-.12,.015,.024,x,top-.102,z+f.height/2-.03);
        }
        if (f.type==="fume_hood" && !equipment.some(item => item.kind === "fume_hood" &&
            item.fixture_id === f.id && [item.x_m, item.y_m].every(Number.isFinite))) {
          const width = Math.min(1.7,f.width);
          const hoodX = x-f.width/2+width/2;
          box(content,width,.14,f.height,0x9db8cb,hoodX,top+1.04,z);
          for (const side of [-1,1]) {
            box(content,.07,1.06,f.height,0x7794aa,hoodX+side*(width-.07)/2,top+.53,z);
          }
          box(content,width-.14,.52,.035,0x83d4ef,hoodX,top+.72,z+f.height/2-.035,.27);
          box(content,width-.16,.022,.04,0xa6e7ff,hoodX,top+.96,z+.09);
        }
      }
      for (const e of equipment) {
        if (![e.x_m,e.y_m].every(Number.isFinite) || !heights.has(e.fixture_id)) continue;
        const x=e.x_m-centerX;
        const z=(isTallCabinet(e.kind) && e.fixture_id === "wall_cabinets" ? wallCabinetCenter : e.y_m)-centerZ;
        const y=isFloorEquipment(e.kind) ? 0 : heights.get(e.fixture_id);
        const top = instrumentModel(e.kind,x,y,z);
        tag(e.name,x,Math.max(y+.57,top+.12),z,"equipment",e.instrument_id);
      }
      const placedCabinets = equipment.filter(item => isTallCabinet(item.kind) &&
        [item.x_m,item.y_m].every(Number.isFinite));
      for (const cabinet of placedCabinets.filter(item => item.kind === "reagent_cabinet")) {
        const chemical = placedCabinets.find(item => item.kind === "chemical_cabinet" &&
          item.fixture_id === cabinet.fixture_id && Math.abs(item.y_m-cabinet.y_m) <= .25);
        if (!chemical) continue;
        const gap = Math.abs(chemical.x_m-cabinet.x_m)-tallCabinetWidth;
        if (gap <= 0 || gap > .8) continue;
        const x = (chemical.x_m+cabinet.x_m)/2-centerX;
        const z = (cabinet.fixture_id === "wall_cabinets" ? wallCabinetCenter
          : (chemical.y_m+cabinet.y_m)/2)-centerZ;
        box(content,gap+.02,1.9,.52,0xa9b6bf,x,.95,z);
      }
      for (const person of people) {
        const x=person.x-centerX,z=person.y-centerZ;
        const history = trails.get(person.track_id) || [];
        const trail = [];
        for (let i=1; i<history.length; i++) {
          const a=history[i-1], b=history[i];
          if (b.breakBefore || ![a.x,a.y,b.x,b.y].every(Number.isFinite)) continue;
          trail.push(a.x-centerX,.035,a.y-centerZ,b.x-centerX,.035,b.y-centerZ);
        }
        drawLines(trail,"trail");
        const previous = history.at(-2), latest = history.at(-1);
        const heading = previous && latest && !latest.breakBefore
          ? Math.atan2(latest.x-previous.x,latest.y-previous.y) : 0;
        personModel(x,z,person.identified,heading);
        trackingOutline(x,z,person.identified);
        tag(personLabel(person),x,1.68,z,person.identified ? "identified" : "unknown");
      }
      draw();
    }
    function setCamera(next) {
      ({yaw,pitch,zoom}=next);
      const instrument = tags.find(tag => tag.id && tag.id === next.instrument);
      if (instrument) target.copy(instrument.position).y -= .43;
      else target.set(0,.35,0);
      draw();
    }
    function instrumentBounds() {
      return tags.filter(tag => tag.id && !tag.node.hidden).map(({id,node}) => {
        const {x,y,width,height} = node.getBoundingClientRect();
        return [id,...[x,y,width,height].map(Math.round)];
      });
    }
    return {root,update,setCamera,instrumentBounds, isReady: () => !contextLost};
  }

  function render(host,map,equipment,personLabel,trails) {
    if (unavailable) return false;
    if (!view) {
      try { view=create(host); }
      catch (error) {
        console.warn("3D map unavailable",error);
        unavailable=true;
        return false;
      }
    }
    if (!view.isReady()) return false;
    try {
      view.update(map,equipment,personLabel,trails);
      return true;
    } catch (error) {
      view.root.hidden = true;
      unavailable = true;
      console.warn("3D map rendering failed", error);
      return false;
    }
  }
  function hide() { if (view) view.root.hidden=true; }
  function setCamera(camera) { view?.setCamera(camera); }
  function instrumentBounds() { return view?.instrumentBounds() || []; }
  return {render,hide,setCamera,instrumentBounds};
})();
