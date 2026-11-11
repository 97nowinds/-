/* Orthographic projection of simplified equipment and people in map metres. */
const LabTwin = (() => {
  const ns = "http://www.w3.org/2000/svg";
  function element(tag, attrs, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text) node.textContent = text;
    return node;
  }

  function render(layer, map, equipment, labelPerson) {
    const scale = Math.min(1010 / (map.width_m + map.height_m), 430 / (.46 * (map.width_m + map.height_m)));
    const origin = 600 - (map.width_m - map.height_m) * scale / 2;
    const project = (x, y, z = 0) => [origin + (x - y) * scale, 153 + (.46 * (x + y) - z) * scale];
    const scene = document.createDocumentFragment();
    const annotations = element("g", {});
    const defs = element("defs", {});
    const floor = element("linearGradient", {id:"twin-floor", x1:"0%", y1:"0%", x2:"100%", y2:"100%"});
    floor.append(element("stop", {offset:"0%", "stop-color":"#304558"}),
      element("stop", {offset:"100%", "stop-color":"#192b3b"}));
    defs.append(floor);
    scene.append(defs);
    const polygon = (points, fill, stroke = "#ffffff15") => scene.append(element("polygon", {
      points: points.map(point => project(...point).join(",")).join(" "), fill, stroke, "stroke-width": 1,
    }));
    function box(x, y, z, w, d, h, colors) {
      const a = x + w, b = y + d, top = z + h;
      polygon([[x,b,z],[a,b,z],[a,b,top],[x,b,top]], colors[1]);
      polygon([[a,y,z],[a,b,z],[a,b,top],[a,y,top]], colors[2]);
      polygon([[x,y,top],[a,y,top],[a,b,top],[x,b,top]], colors[0]);
    }
    function badge(px, py, text, color) {
      const caption = text.length > 20 ? text.slice(0, 19) + "…" : text;
      const width = Math.max(84, Array.from(caption).reduce((sum, c) =>
        sum + (c.charCodeAt(0) > 127 ? 17 : 9), 34));
      const x = Math.max(12, Math.min(1188-width, px-width/2));
      annotations.append(element("rect", {x, y:py-26, width, height:31, rx:9,
        fill:"#111f2d", stroke:"#537386"}));
      annotations.append(element("circle", {cx:x+13, cy:py-10, r:3.5, fill:color}));
      annotations.append(element("text", {x:x+24, y:py-5, fill:"#f2f7fa",
        class:"twin-label"}, caption));
    }
    scene.append(element("ellipse", {cx:600, cy:480, rx:450, ry:107, fill:"#030812",opacity:.36}));
    box(0, 0, -.25, map.width_m, map.height_m, .25, ["url(#twin-floor)","#0b1522","#142433"]);
    for (const [limit, vertical] of [[map.width_m, true], [map.height_m, false]]) {
      for (let i = 2; i < limit; i += 2) {
        const [x1,y1] = vertical ? project(i,0) : project(0,i);
        const [x2,y2] = vertical ? project(i,map.height_m) : project(map.width_m,i);
        scene.append(element("line",{x1,y1,x2,y2,stroke:"#7695aa","stroke-opacity":.16}));
      }
    }

    const items = [];
    const benches = new Map();
    for (const f of map.fixtures || []) {
      if (f.type === "door") continue;
      const height = f.type === "cabinet" ? 1.05 : .86;
      benches.set(f.id, height);
      items.push({depth:f.x + f.width + f.y + f.height, bench:true, draw() {
        if (f.type === "bench") {
          const bays = Math.max(1, Math.ceil(f.width/2.5));
          for (let i=0; i<=bays; i++) {
            const x = f.x + .08 + i*(f.width-.26)/bays;
            for (const y of [f.y+.08,f.y+f.height-.18]) {
              box(x,y,0,.10,.10,height-.10,["#7696a4","#66828f","#3e5a6a"]);
            }
          }
          box(f.x+.10,f.y+f.height-.15,height-.30,f.width-.20,.10,.20,["#74929f","#537383","#425f70"]);
        } else {
          box(f.x+.08,f.y+.08,.06,f.width-.16,f.height-.16,height-.16,["#a3b4bd","#6f8796","#496273"]);
        }
        box(f.x,f.y,height-.10,f.width,f.height,.10,["#9eb2bb","#6b8999","#4d6b7c"]);
        const [x,y] = project(f.x+f.width/2,f.y+f.height,height-.28);
        scene.append(element("text",{x,y,fill:"#e0e9ed","text-anchor":"middle",class:"twin-fixture-label"},f.name));
      }});
    }
    for (const item of equipment) {
      const {x_m:x, y_m:y} = item;
      if (![x,y].every(Number.isFinite) || !benches.has(item.fixture_id)) continue;
      const z = benches.get(item.fixture_id);
      items.push({depth:x+y+.45, draw() {
        box(x-.25,y-.20,z,.50,.40,.15,["#d5e2e6","#7899a7","#566f80"]);
        if (item.kind === "balance") {
          box(x-.18,y-.15,z+.15,.36,.30,.36,["#84dbe866","#5ebdd34d","#429bb660"]);
          box(x-.11,y-.08,z+.15,.22,.16,.035,["#e1faff","#8fc4d1","#6194a3"]);
        } else if (item.kind === "cleaner") {
          box(x-.24,y-.19,z+.15,.48,.38,.24,["#c0d4db","#7897a3","#506e7d"]);
          polygon([[x-.17,y-.12,z+.391],[x+.17,y-.12,z+.391],[x+.17,y+.12,z+.391],
            [x-.17,y+.12,z+.391]],"#203c51","#88c5d5");
        } else {
          box(x-.23,y-.18,z+.15,.46,.36,.28,["#e2dbc6","#a8a28f","#7e7e73"]);
          box(x-.17,y-.13,z+.43,.34,.26,.045,["#ede5cd","#afa88f","#97937f"]);
          box(x-.13,y+.15,z+.25,.26,.041,.085,["#33728f","#17384d","#20475e"]);
        }
        const offset = item.kind === "balance" ? .95 : item.kind === "cleaner" ? -.95 : 0;
        const [x1,y1] = project(x,y,z+.45);
        const [x2,y2] = project(x+offset,y+offset,z+1.1);
        const badgeX = x2+offset*83;
        annotations.append(element("path",{d:`M${x1} ${y1} L${x2} ${y2} H${badgeX}`,
          fill:"none",stroke:"#82b7c6","stroke-opacity":.82,"stroke-width":1.3}));
        annotations.append(element("circle",{cx:x1,cy:y1,r:3,fill:"#a4e3e9"}));
        badge(badgeX,y2,item.name,"#a4e3e9");
      }});
    }
    for (const person of map.people || []) {
      const {x,y} = person;
      if (person.map_visible === false || ![x,y].every(Number.isFinite)) continue;
      items.push({depth:x+y, draw() {
        const color = person.identified ? "#79dac0" : "#f0b67b";
        const [cx,cy] = project(x,y);
        scene.append(element("ellipse",{cx,cy,rx:13,ry:6,fill:color,opacity:.22}));
        for (const offset of [-.105,.015]) {
          box(x+offset,y-.07,0,.09,.14,.36,["#567389","#304e65","#233c51"]);
        }
        box(x-.17,y-.10,.36,.34,.20,.42,[color,person.identified ? "#438d81" : "#b67b52","#405d73"]);
        for (const offset of [-.21,.15]) {
          box(x+offset,y-.06,.32,.065,.12,.36,[color,"#487b83","#2e5666"]);
        }
        const [hx,hy] = project(x,y,.88);
        scene.append(element("circle",{cx:hx,cy:hy,r:8,fill:"#e9c7ae",stroke:"#233b4c"}));
        scene.append(element("path",{d:`M${hx-7} ${hy-2} a7 7 0 0 1 14 0`,fill:"#263545"}));
        badge(hx,hy-20,labelPerson(person),color);
      }});
    }
    items.sort((a,b) => Number(Boolean(b.bench))-Number(Boolean(a.bench)) || a.depth-b.depth);
    for (const item of items) item.draw();
    scene.append(annotations);
    layer.replaceChildren(scene);
  }
  return {render};
})();
