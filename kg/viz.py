from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import audit, export, observations


def visualization_dict(conn: sqlite3.Connection) -> dict:
    payload = export.graph_dict(conn)
    payload["graph_audit"] = audit.graph_report(conn)
    payload["rejection_audit"] = audit.rejection_report(conn)
    payload["observation_audit"] = observations.observation_audit(conn)
    return payload


def write_html(conn: sqlite3.Connection, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        visualization_dict(conn),
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = _HTML.replace("__GRAPH_DATA__", data)
    output.write_text(html, encoding="utf-8")
    return output


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 知识图谱审计视图</title>
<style>
:root{color-scheme:dark;--bg:#080d16;--panel:#0f1725;--panel2:#141f30;--line:#233148;--text:#edf4ff;--muted:#91a1b8;--accent:#5cc8ff;--shadow:0 18px 44px #0006}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 35% 20%,#14243a 0,#080d16 48%);color:var(--text);font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;overflow:hidden}
header{height:72px;display:flex;gap:14px;align-items:center;padding:12px 18px;border-bottom:1px solid #26354a;background:#0b121eee;backdrop-filter:blur(14px)}
.brand{display:flex;align-items:center;gap:10px;min-width:max-content}.brand-mark{width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,#5cc8ff,#9478ff);box-shadow:0 0 24px #5cc8ff55;position:relative}.brand-mark:after{content:"";position:absolute;inset:8px;border:2px solid white;border-radius:50%}
header strong{display:block;font-size:16px;letter-spacing:.02em}.subtitle{font-size:11px;color:var(--muted)}.stats{color:#b8c7da;white-space:nowrap;padding-left:14px;border-left:1px solid var(--line)}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-left:auto}
input,select,button{font:inherit;background:#141f30;color:var(--text);border:1px solid #2b3b53;border-radius:8px;padding:7px 9px;outline:none;transition:.18s border-color,.18s background,.18s transform}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px #5cc8ff1c}button{cursor:pointer}button:hover{border-color:#5cc8ff;background:#192943}button:active{transform:translateY(1px)}
.search-wrap{position:relative}.search-wrap input{width:210px;padding-left:30px}.search-wrap:before{content:"⌕";position:absolute;left:10px;top:5px;color:var(--muted);font-size:18px}
label{color:var(--muted);font-size:12px}label input{vertical-align:middle;accent-color:var(--accent)}
main{display:grid;grid-template-columns:minmax(0,1fr) 410px;height:calc(100vh - 72px)}
.stage{position:relative;overflow:hidden;background-image:radial-gradient(#29405b80 1px,transparent 1px);background-size:24px 24px}.stage:after{content:"";pointer-events:none;position:absolute;inset:0;box-shadow:inset 0 0 90px #050911aa}
canvas{width:100%;height:100%;display:block;cursor:grab}.hint,.legend{position:absolute;z-index:2;color:var(--muted);background:#0c1421e8;border:1px solid #273750;box-shadow:var(--shadow);backdrop-filter:blur(10px);border-radius:10px}
.hint{left:16px;bottom:14px;padding:7px 11px;font-size:12px}.legend{left:16px;top:16px;padding:10px 12px;display:grid;gap:7px}.legend-row{display:flex;align-items:center;gap:7px;font-size:12px}.legend-line{width:22px;height:2px;border-radius:2px}.legend-dot{width:9px;height:9px;border-radius:50%}
.graph-actions{position:absolute;z-index:3;right:14px;top:14px;display:flex;gap:7px}.graph-actions button{width:36px;height:34px;padding:0;font-size:17px;background:#0c1421e8;box-shadow:var(--shadow)}
.tooltip{position:absolute;z-index:4;pointer-events:none;display:none;max-width:260px;padding:8px 10px;border:1px solid #32465f;border-radius:8px;background:#0a111ddd;box-shadow:var(--shadow);font-size:12px}.tooltip b{display:block;color:#fff;margin-bottom:2px}.tooltip span{color:var(--muted)}
aside{overflow:auto;border-left:1px solid var(--line);background:linear-gradient(180deg,#111b2a,#0d1521);padding:18px;scrollbar-color:#33445d transparent}
.section-title{display:flex;align-items:center;justify-content:space-between;margin:0 0 10px}.section-title h2{margin:0}.eyebrow{color:var(--accent);font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
h2{font-size:14px;margin:15px 0 8px;color:#d9e8fb}pre{white-space:pre-wrap;word-break:break-word;margin:0;color:#c8d7e9;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}
#details{min-height:90px;max-height:34vh;overflow:auto;padding:12px;border:1px solid #263750;border-radius:10px;background:#0a121e}
.badge{display:inline-block;padding:3px 8px;border:1px solid #2d425f;border-radius:999px;background:#17263a;margin:3px 3px 3px 0;color:#cfe0f5;font-size:11px}
.audit-section{border-top:1px solid var(--line);margin-top:16px;padding-top:13px}.sample{padding:8px 0;border-bottom:1px solid #203149;color:#b9c9dc;font-size:12px}
.audit-controls{display:grid;grid-template-columns:1fr 145px;gap:7px;margin:9px 0}.audit-controls input,.audit-controls select{min-width:0;width:100%}
.obs-list{max-height:340px;overflow:auto}.obs-item{display:block;width:100%;text-align:left;margin:6px 0;padding:9px 10px;border-color:#2a405f;color:var(--text);background:#121e2e}
.obs-item small{display:block;color:var(--muted);margin-top:4px}.obs-item:hover{background:#192a41;transform:translateX(2px)}.empty{color:var(--muted);padding:8px 0;font-size:12px}
.status-blocked{border-left:3px solid #ff6b89}.status-pending_judgment{border-left:3px solid #ffc857}.status-pending_endpoint{border-left:3px solid #5cc8ff}.status-supported_unmaterialized{border-left:3px solid #b995ff}.status-insufficient,.status-contradicts{border-left:3px solid #ff9f9f}
@media(max-width:1080px){.stats{display:none}.controls label{display:none}}@media(max-width:820px){body{overflow:auto}header{height:auto;align-items:flex-start}.controls{justify-content:flex-end}.search-wrap input{width:170px}main{grid-template-columns:1fr;grid-template-rows:68vh auto;height:auto}aside{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header>
  <div class="brand"><span class="brand-mark"></span><div><strong>AI 知识图谱</strong><div class="subtitle">Corpus-grounded graph explorer</div></div></div>
  <span class="stats" id="stats"></span>
  <div class="controls">
    <div class="search-wrap"><input id="search" placeholder="搜索实体，回车聚焦"></div>
    <label>层数 <select id="hops"><option>1</option><option selected>2</option><option>3</option></select></label>
    <label>节点 <select id="limit"><option selected>60</option><option>120</option><option>200</option></select></label>
    <label><input class="rel" type="checkbox" value="is_a" checked>is_a</label>
    <label><input class="rel" type="checkbox" value="part_of" checked>part_of</label>
    <label><input class="rel" type="checkbox" value="prerequisite_of" checked>prerequisite</label>
    <button id="reset">核心节点</button>
  </div>
</header>
<main>
  <section class="stage">
    <canvas id="graph"></canvas>
    <div class="legend">
      <div class="legend-row"><i class="legend-line" style="background:#5cc8ff"></i>is_a 分类</div>
      <div class="legend-row"><i class="legend-line" style="background:#ffc857"></i>part_of 组成 / 归属</div>
      <div class="legend-row"><i class="legend-line" style="background:#ff6b89"></i>prerequisite 先修</div>
      <div class="legend-row"><i class="legend-dot" style="background:#a7e37d"></i>节点颜色表示 Entity 类型</div>
    </div>
    <div class="graph-actions"><button id="fit" title="适应窗口">⌗</button><button id="reheat" title="重新布局">↻</button></div>
    <div id="tooltip" class="tooltip"></div>
    <div class="hint">拖拽节点 · 拖动画布 · 滚轮缩放 · 点击查看 Evidence</div>
  </section>
  <aside>
    <div class="section-title"><div><div class="eyebrow">Inspector</div><h2>选中对象</h2></div><span class="badge" id="selection-kind">未选择</span></div>
    <pre id="details">点击节点或边查看定义、类型和 Evidence。</pre>
    <section class="audit-section">
      <h2>Observation 审计</h2>
      <div id="obs-summary"></div>
      <div class="audit-controls">
        <input id="obs-search" placeholder="搜索待定端点或原文">
        <select id="obs-status">
          <option value="all">全部未落实</option>
          <option value="pending_endpoint">待定端点</option>
          <option value="pending_judgment">待裁判</option>
          <option value="supported_unmaterialized">支持但未落实</option>
          <option value="blocked">物化阻塞</option>
          <option value="insufficient">insufficient</option>
          <option value="contradicts">contradicts</option>
        </select>
      </div>
      <div id="promotion-candidates"></div>
      <div id="obs-count" class="empty"></div>
      <div id="obs-list" class="obs-list"></div>
    </section>
    <section class="audit-section">
      <h2>历史拒绝审计</h2>
      <div id="reject-summary"></div>
      <div id="reject-samples"></div>
    </section>
  </aside>
</main>
<script id="graph-data" type="application/json">__GRAPH_DATA__</script>
<script>
const data=JSON.parse(document.getElementById("graph-data").textContent);
const oa=data.observation_audit||{summary:{},items:[],promotion_candidates:[]};
const byId=new Map(data.entities.map(x=>[x.id,x]));
const evidenceByTarget=new Map();
for(const e of data.evidence){if(!evidenceByTarget.has(e.target))evidenceByTarget.set(e.target,[]);evidenceByTarget.get(e.target).push(e)}
const degree=new Map(data.entities.map(x=>[x.id,0]));
for(const e of data.claims){degree.set(e.subject_id,(degree.get(e.subject_id)||0)+1);degree.set(e.object_id,(degree.get(e.object_id)||0)+1)}
const colors={is_a:"#5cc8ff",part_of:"#ffc857",prerequisite_of:"#ff6b89"};
const typeColors={resource:"#b995ff",criterion:"#ff9f9f",data:"#70d7e5",task:"#ffc857",solution:"#5cc8ff",concept:"#a7e37d"};
let root=[...degree.entries()].sort((a,b)=>b[1]-a[1])[0]?.[0]||data.entities[0]?.id;
let view={nodes:[],edges:[],positions:new Map()},selected=null,hovered=null,alpha=0,frame=0;
let transform={x:0,y:0,scale:1},drag=null,nodeDrag=null;
const canvas=document.getElementById("graph"),ctx=canvas.getContext("2d");
function activeRelations(){return new Set([...document.querySelectorAll(".rel:checked")].map(x=>x.value))}
function primaryType(entity){return entity.type_profile?.[0]?.entity_type||"concept"}
function build(){
 const allowed=activeRelations(),adj=new Map(data.entities.map(x=>[x.id,[]]));
 for(const edge of data.claims)if(allowed.has(edge.relation)){adj.get(edge.subject_id).push([edge.object_id,edge]);adj.get(edge.object_id).push([edge.subject_id,edge])}
 const max=+document.getElementById("limit").value,hops=+document.getElementById("hops").value;
 const queue=[[root,0]],seen=new Set([root]),levels=new Map([[root,0]]);
 while(queue.length&&seen.size<max){const [id,level]=queue.shift();if(level>=hops)continue;
   const next=(adj.get(id)||[]).slice().sort((a,b)=>(degree.get(b[0])||0)-(degree.get(a[0])||0));
   for(const [other] of next)if(!seen.has(other)){seen.add(other);levels.set(other,level+1);queue.push([other,level+1]);if(seen.size>=max)break}
 }
 view.nodes=[...seen].map(id=>byId.get(id));view.edges=data.claims.filter(e=>allowed.has(e.relation)&&seen.has(e.subject_id)&&seen.has(e.object_id));
 layout(levels);startSimulation();document.getElementById("stats").textContent=`当前 ${view.nodes.length} 节点 / ${view.edges.length} 边 · 全库 ${data.entities.length}/${data.claims.length} · Observation ${oa.summary.observations||0} · 待定 ${oa.summary.pending_endpoint||0}`;
}
function layout(levels){
 const w=canvas.clientWidth,h=canvas.clientHeight,cx=w/2,cy=h/2;view.positions=new Map();
 for(const n of view.nodes){const level=levels.get(n.id)||0,angle=(n.id*2.399963+level*.71)%(Math.PI*2),radius=level?Math.min(w,h)*(.09+level*.13):0;view.positions.set(n.id,{x:cx+Math.cos(angle)*radius,y:cy+Math.sin(angle)*radius,vx:0,vy:0,fixed:false})}
 transform={x:0,y:0,scale:1};
}
function startSimulation(){alpha=1;cancelAnimationFrame(frame);frame=requestAnimationFrame(tick)}
function tick(){simulate();draw();if(alpha>.015)frame=requestAnimationFrame(tick)}
function simulate(){
 const nodes=view.nodes,pos=view.positions,w=canvas.clientWidth,h=canvas.clientHeight;
 for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=pos.get(nodes[i].id),b=pos.get(nodes[j].id),dx=b.x-a.x||.1,dy=b.y-a.y||.1,d2=Math.max(100,dx*dx+dy*dy),f=Math.min(1.8,900/d2)*alpha,fx=dx*f/Math.sqrt(d2),fy=dy*f/Math.sqrt(d2);a.vx-=fx;a.vy-=fy;b.vx+=fx;b.vy+=fy}
 for(const edge of view.edges){const a=pos.get(edge.subject_id),b=pos.get(edge.object_id),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-105)*.006*alpha,fx=dx/d*f,fy=dy/d*f;a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy}
 for(const n of nodes){const p=pos.get(n.id);if(p.fixed)continue;p.vx+=(w/2-p.x)*.0008*alpha;p.vy+=(h/2-p.y)*.0008*alpha;p.vx*=.84;p.vy*=.84;p.x+=p.vx;p.y+=p.vy}
 alpha*=.965;
}
function resize(){const d=devicePixelRatio||1;canvas.width=canvas.clientWidth*d;canvas.height=canvas.clientHeight*d;draw()}
function screen(p){return{x:p.x*transform.scale+transform.x,y:p.y*transform.scale+transform.y}}
function nodeRadius(n){return Math.max(5,Math.min(12,5+Math.sqrt(degree.get(n.id)||0)*1.15))}
function relatedIds(item){const ids=new Set();if(!item)return ids;if(item.canonical_name){ids.add(item.id);for(const e of view.edges){if(e.subject_id===item.id)ids.add(e.object_id);if(e.object_id===item.id)ids.add(e.subject_id)}}else{ids.add(item.subject_id);ids.add(item.object_id)}return ids}
function draw(){
 const d=devicePixelRatio||1;ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);const focus=relatedIds(hovered||selected);
 for(const edge of view.edges){const ap=screen(view.positions.get(edge.subject_id)),bp=screen(view.positions.get(edge.object_id)),an=byId.get(edge.subject_id),bn=byId.get(edge.object_id),ar=nodeRadius(an)*transform.scale,br=nodeRadius(bn)*transform.scale,dx=bp.x-ap.x,dy=bp.y-ap.y,len=Math.max(1,Math.hypot(dx,dy)),x1=ap.x+dx/len*ar,y1=ap.y+dy/len*ar,x2=bp.x-dx/len*(br+5),y2=bp.y-dy/len*(br+5),active=!focus.size||(focus.has(edge.subject_id)&&focus.has(edge.object_id));ctx.globalAlpha=active ? .72 : .1;ctx.strokeStyle=colors[edge.relation];ctx.fillStyle=colors[edge.relation];ctx.lineWidth=edge===selected?2.6:1.25;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();const a=Math.atan2(dy,dx),s=5;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-Math.cos(a-.48)*s,y2-Math.sin(a-.48)*s);ctx.lineTo(x2-Math.cos(a+.48)*s,y2-Math.sin(a+.48)*s);ctx.closePath();ctx.fill()}
 for(const node of view.nodes){const p=screen(view.positions.get(node.id)),r=nodeRadius(node)*transform.scale,isFocus=!focus.size||focus.has(node.id),isSelected=node===selected||node.id===root;ctx.globalAlpha=isFocus?1:.18;ctx.fillStyle=typeColors[primaryType(node)]||"#a7e37d";ctx.shadowColor=isSelected?ctx.fillStyle:"transparent";ctx.shadowBlur=isSelected?15:0;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle=isSelected?"#fff":"#07101b";ctx.lineWidth=isSelected?2:1.2;ctx.stroke();
   /* All visible nodes keep their names. A dark outline separates labels from edges. */
   const fontSize=Math.max(10,Math.min(13,11.5*transform.scale));ctx.font=`600 ${fontSize}px system-ui`;ctx.lineJoin="round";ctx.lineWidth=4;ctx.strokeStyle="#07101be8";ctx.strokeText(node.canonical_name,p.x+r+5,p.y+fontSize*.35);ctx.fillStyle=isFocus?"#edf4ff":"#8190a4";ctx.fillText(node.canonical_name,p.x+r+5,p.y+fontSize*.35)
 }ctx.globalAlpha=1;
}
function nodeAt(x,y){let best=null,dist=20;for(const n of view.nodes){const p=screen(view.positions.get(n.id)),d=Math.hypot(x-p.x,y-p.y);if(d<dist){best=n;dist=d}}return best}
function edgeAt(x,y){let best=null,dist=7;for(const e of view.edges){const a=screen(view.positions.get(e.subject_id)),b=screen(view.positions.get(e.object_id));const dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((x-a.x)*dx+(y-a.y)*dy)/(dx*dx+dy*dy||1)));const d=Math.hypot(x-(a.x+t*dx),y-(a.y+t*dy));if(d<dist){best=e;dist=d}}return best}
function showNode(n){selected=n;document.getElementById("selection-kind").textContent="Entity";const ev=evidenceByTarget.get(`entity:${n.id}`)||[];document.getElementById("details").textContent=[
 n.canonical_name,`ID: ${n.id}  degree: ${degree.get(n.id)||0}`,`aliases: ${(n.aliases||[]).join(" / ")}`,`definition: ${n.definition}`,
 `type_profile: ${JSON.stringify(n.type_profile)}`,...ev.slice(0,6).map((e,i)=>`\\nEvidence ${i+1} · ${e.source.name} · ${e.location}\\n原文: ${e.source_text}\\n模型引文: ${e.model_quote}`)
 ].join("\\n");draw()}
function showEdge(e){selected=e;document.getElementById("selection-kind").textContent="Claim";const ev=evidenceByTarget.get(`claim:${e.id}`)||[];document.getElementById("details").textContent=[
 `${e.subject}  --${e.relation}-->  ${e.object}`,`Claim ID: ${e.id}`,...ev.map((x,i)=>`\\nEvidence ${i+1} · ${x.source.name} · ${x.location}\\n原文: ${x.source_text}\\n裁判: ${x.validation.verdict} · ${x.validation.reason}`)
 ].join("\\n");draw()}
const statusLabels={pending_endpoint:"待定端点",pending_judgment:"待裁判",supported_unmaterialized:"支持但未落实",blocked:"物化阻塞",insufficient:"证据不足",contradicts:"证据矛盾"};
function endpointText(endpoint){return endpoint.entity_id?`${endpoint.name} → ${endpoint.entity_name} (#${endpoint.entity_id})`:`${endpoint.name} → 未解析`}
function showObservation(item){selected=null;document.getElementById("selection-kind").textContent="Observation";document.getElementById("details").textContent=[
 `ClaimObservation #${item.id} · ${(item.statuses||[item.status]).map(x=>statusLabels[x]||x).join(" / ")}`,
 `${endpointText(item.subject)}\\n  --${item.relation} / ${item.polarity}-->\\n${endpointText(item.object)}`,
 `Source: ${item.source.name} (#${item.source.id}) · Chunk ${item.chunk_index}`,
 `Passage: ${(item.passage_ids||[]).join(", ")} · ${item.location}`,
 `裁判: ${item.validation.verdict||"尚未裁判"} · ${item.validation.reason||""}`,
 item.materialization_error?`物化错误: ${item.materialization_error}`:"",
 `抽取: ${item.extraction.model} / ${item.extraction.prompt_version}`,
 `裁判版本: ${item.validation.model} / ${item.validation.prompt_version}`,
 `\\n真实原文:\\n${item.source_text}`,
 `\\n模型引文:\\n${item.model_quote}`
 ].filter(Boolean).join("\\n")}
function renderObservations(){
 const q=document.getElementById("obs-search").value.trim().toLowerCase(),status=document.getElementById("obs-status").value;
 const filtered=oa.items.filter(x=>(status==="all"||(x.statuses||[x.status]).includes(status))&&(!q||[x.subject.name,x.object.name,x.relation,x.source.name,x.source_text,x.model_quote].join(" ").toLowerCase().includes(q)));
 document.getElementById("obs-count").textContent=`显示 ${Math.min(filtered.length,100)} / ${filtered.length} 条；页面载入 ${oa.items.length} 条未落实记录`;
 const box=document.getElementById("obs-list");box.replaceChildren();
 if(!filtered.length){const empty=document.createElement("div");empty.className="empty";empty.textContent="没有符合条件的 Observation";box.appendChild(empty);return}
 for(const item of filtered.slice(0,100)){const button=document.createElement("button");button.className=`obs-item status-${item.status}`;const title=document.createElement("span");title.textContent=`${item.subject.name}  --${item.relation}-->  ${item.object.name}`;const meta=document.createElement("small");meta.textContent=`${(item.statuses||[item.status]).map(x=>statusLabels[x]||x).join(" / ")} · ${item.source.name} · Chunk ${item.chunk_index}`;button.append(title,meta);button.onclick=()=>showObservation(item);box.appendChild(button)}
}
function renderObservationAudit(){
 const s=oa.summary;document.getElementById("obs-summary").innerHTML=`<span class="badge">总计 ${s.observations||0}</span><span class="badge">已落实 ${s.materialized||0}</span><span class="badge">待定端点 ${s.pending_endpoint||0}</span><span class="badge">待裁判 ${s.pending_judgment||0}</span><span class="badge">支持未落实 ${s.supported_unmaterialized||0}</span><span class="badge">晋升候选 ${s.promotion_candidates_3plus||0}</span>`;
 const candidateBox=document.getElementById("promotion-candidates");candidateBox.replaceChildren();
 for(const candidate of oa.promotion_candidates){const button=document.createElement("button");button.className="obs-item status-pending_endpoint";button.textContent=`晋升候选：${candidate.name} · ${candidate.passage_count} passages / ${candidate.source_count} sources`;button.onclick=()=>{document.getElementById("details").textContent=[`Entity 晋升候选：${candidate.name}`,`变体: ${candidate.names.join(" / ")}`,`${candidate.passage_count} 个独立 Passage · ${candidate.source_count} 个 Source`,...candidate.evidence.map((x,i)=>`\\n证据 ${i+1} · Source #${x.source_id} · ${x.passage_ids.join(", ")}\\n${x.source_text}`)].join("\\n")};candidateBox.appendChild(button)}
 renderObservations();
}
canvas.addEventListener("click",e=>{if(drag?.moved||nodeDrag?.moved)return;const n=nodeAt(e.offsetX,e.offsetY);if(n){showNode(n);return}const edge=edgeAt(e.offsetX,e.offsetY);if(edge){showEdge(edge);return}selected=null;document.getElementById("selection-kind").textContent="未选择";draw()});
canvas.addEventListener("mousedown",e=>{const n=nodeAt(e.offsetX,e.offsetY);if(n){const p=view.positions.get(n.id);p.fixed=true;nodeDrag={node:n,x:e.clientX,y:e.clientY,moved:false};canvas.style.cursor="grabbing"}else{drag={x:e.clientX,y:e.clientY,ox:transform.x,oy:transform.y,moved:false};canvas.style.cursor="grabbing"}});
addEventListener("mousemove",e=>{if(nodeDrag){const p=view.positions.get(nodeDrag.node.id),rect=canvas.getBoundingClientRect();nodeDrag.moved=nodeDrag.moved||Math.abs(e.clientX-nodeDrag.x)+Math.abs(e.clientY-nodeDrag.y)>3;p.x=(e.clientX-rect.left-transform.x)/transform.scale;p.y=(e.clientY-rect.top-transform.y)/transform.scale;p.vx=p.vy=0;draw();return}if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved=Math.abs(dx)+Math.abs(dy)>3;transform.x=drag.ox+dx;transform.y=drag.oy+dy;draw()});
addEventListener("mouseup",()=>{drag=null;nodeDrag=null;canvas.style.cursor="grab"});
canvas.addEventListener("mousemove",e=>{if(drag||nodeDrag)return;const hit=nodeAt(e.offsetX,e.offsetY),tip=document.getElementById("tooltip");hovered=hit;canvas.style.cursor=hit?"pointer":"grab";if(hit){tip.style.display="block";tip.style.left=`${Math.min(e.offsetX+14,canvas.clientWidth-270)}px`;tip.style.top=`${Math.max(12,e.offsetY-18)}px`;tip.innerHTML=`<b>${hit.canonical_name}</b><span>${primaryType(hit)} · degree ${degree.get(hit.id)||0}</span>`}else tip.style.display="none";draw()});
canvas.addEventListener("mouseleave",()=>{hovered=null;document.getElementById("tooltip").style.display="none";draw()});
canvas.addEventListener("wheel",e=>{e.preventDefault();const factor=e.deltaY<0?1.12:.89,old=transform.scale;transform.scale=Math.max(.3,Math.min(4,old*factor));transform.x=e.offsetX-(e.offsetX-transform.x)*transform.scale/old;transform.y=e.offsetY-(e.offsetY-transform.y)*transform.scale/old;draw()},{passive:false});
document.getElementById("search").addEventListener("keydown",e=>{if(e.key!=="Enter")return;const q=e.target.value.trim().toLowerCase();const hit=data.entities.find(x=>x.canonical_name.toLowerCase()===q)||data.entities.find(x=>x.canonical_name.toLowerCase().includes(q)||(x.aliases||[]).some(a=>a.toLowerCase().includes(q)));if(hit){root=hit.id;build();showNode(hit)}});
document.getElementById("reset").onclick=()=>{root=[...degree.entries()].sort((a,b)=>b[1]-a[1])[0][0];build()};
document.getElementById("fit").onclick=()=>{transform={x:0,y:0,scale:1};draw()};
document.getElementById("reheat").onclick=()=>{for(const p of view.positions.values()){p.fixed=false;p.vx=p.vy=0}startSimulation()};
for(const x of document.querySelectorAll(".rel,#hops,#limit"))x.addEventListener("change",build);
document.getElementById("obs-search").addEventListener("input",renderObservations);
document.getElementById("obs-status").addEventListener("change",renderObservations);
const ga=data.graph_audit,ra=data.rejection_audit;
document.getElementById("reject-summary").innerHTML=`<span class="badge">总计 ${ra.total}</span><span class="badge">算法损失 ${ra.algorithmic_loss}</span><span class="badge">语义拒绝 ${ra.semantic_rejection}</span>`+Object.entries(ra.categories).map(([k,v])=>`<span class="badge">${k}: ${v}</span>`).join("");
const sampleBox=document.getElementById("reject-samples");for(const [category,items] of Object.entries(ra.samples)){const title=document.createElement("h2");title.textContent=category;sampleBox.appendChild(title);for(const item of items.slice(0,5)){const div=document.createElement("div");div.className="sample";div.textContent=`Chunk ${item.chunk}: ${item.message}`;sampleBox.appendChild(div)}}
addEventListener("resize",resize);resize();build();renderObservationAudit();
</script>
</body>
</html>
"""
