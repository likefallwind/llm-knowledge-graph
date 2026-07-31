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
:root{color-scheme:dark;--bg:#09111f;--panel:#111c2e;--line:#263852;--text:#e7eef9;--muted:#91a4bd;--accent:#66d9ef}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif}
header{display:flex;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:#0d1727}
header strong{font-size:17px}.stats{color:var(--muted);white-space:nowrap}
.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-left:auto}
input,select,button{background:#16243a;color:var(--text);border:1px solid #334967;border-radius:6px;padding:7px 9px}
button{cursor:pointer}button:hover{border-color:var(--accent)}
label{color:var(--muted)}label input{vertical-align:middle}
main{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:calc(100vh - 62px)}
.stage{position:relative;overflow:hidden}.hint{position:absolute;left:12px;bottom:10px;color:var(--muted);background:#09111fcc;padding:6px 9px;border-radius:5px}
canvas{width:100%;height:100%;display:block}
aside{overflow:auto;border-left:1px solid var(--line);background:var(--panel);padding:14px}
h2{font-size:15px;margin:14px 0 8px;color:#c9dcf3}pre{white-space:pre-wrap;word-break:break-word;margin:0;color:#cad8ea}
#details{max-height:38vh;overflow:auto;padding-right:5px}
.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:#1d304a;margin:2px 3px 2px 0;color:#cfe0f5}
.audit-section{border-top:1px solid var(--line);margin-top:14px;padding-top:10px}.sample{padding:7px 0;border-bottom:1px solid #203149;color:#b9c9dc}
.audit-controls{display:grid;grid-template-columns:1fr 145px;gap:7px;margin:8px 0}.audit-controls input,.audit-controls select{min-width:0;width:100%}
.obs-list{max-height:360px;overflow:auto}.obs-item{display:block;width:100%;text-align:left;margin:5px 0;padding:8px;border-color:#2a405f;color:var(--text)}
.obs-item small{display:block;color:var(--muted);margin-top:3px}.obs-item:hover{background:#1b2d47}.empty{color:var(--muted);padding:8px 0}
.status-blocked{border-left:3px solid #ef6f91}.status-pending_judgment{border-left:3px solid #f6c177}.status-pending_endpoint{border-left:3px solid #66d9ef}.status-supported_unmaterialized{border-left:3px solid #c4a7e7}.status-insufficient,.status-contradicts{border-left:3px solid #ebbcba}
@media(max-width:900px){main{grid-template-columns:1fr;grid-template-rows:65vh auto;height:auto}aside{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header>
  <strong>AI 知识图谱审计视图</strong>
  <span class="stats" id="stats"></span>
  <div class="controls">
    <input id="search" placeholder="搜索实体，回车定位">
    <label>层数 <select id="hops"><option>1</option><option selected>2</option><option>3</option></select></label>
    <label>节点 <select id="limit"><option>60</option><option selected>120</option><option>200</option></select></label>
    <label><input class="rel" type="checkbox" value="is_a" checked>is_a</label>
    <label><input class="rel" type="checkbox" value="part_of" checked>part_of</label>
    <label><input class="rel" type="checkbox" value="prerequisite_of" checked>prerequisite</label>
    <button id="reset">核心节点</button>
  </div>
</header>
<main>
  <section class="stage">
    <canvas id="graph"></canvas>
    <div class="hint">滚轮缩放 · 拖动画布 · 点击节点或边查看原文证据</div>
  </section>
  <aside>
    <h2>选中对象</h2>
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
const colors={is_a:"#66d9ef",part_of:"#f6c177",prerequisite_of:"#ef6f91"};
const typeColors={resource:"#c4a7e7",criterion:"#ebbcba",data:"#9ccfd8",task:"#f6c177",solution:"#66d9ef",concept:"#a6da95"};
let root=[...degree.entries()].sort((a,b)=>b[1]-a[1])[0]?.[0]||data.entities[0]?.id;
let view={nodes:[],edges:[],positions:new Map()};
let transform={x:0,y:0,scale:1},drag=null;
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
 layout(levels);draw();document.getElementById("stats").textContent=`${view.nodes.length} 节点 / ${view.edges.length} 边（全库 ${data.entities.length}/${data.claims.length} · Observation ${oa.summary.observations||0} · 待定 ${oa.summary.pending_endpoint||0}）`;
}
function layout(levels){
 const groups=new Map();for(const n of view.nodes){const l=levels.get(n.id)||0;if(!groups.has(l))groups.set(l,[]);groups.get(l).push(n)}
 const w=canvas.clientWidth,h=canvas.clientHeight,cx=w/2,cy=h/2;view.positions=new Map();
 for(const [level,nodes] of groups){nodes.sort((a,b)=>(degree.get(b.id)||0)-(degree.get(a.id)||0)||a.id-b.id);
   if(level===0){view.positions.set(nodes[0].id,{x:cx,y:cy});continue}
   const radius=Math.min(w,h)*(.18+level*.16);nodes.forEach((n,i)=>{const a=2*Math.PI*i/nodes.length-Math.PI/2;view.positions.set(n.id,{x:cx+Math.cos(a)*radius,y:cy+Math.sin(a)*radius})})
 }
 transform={x:0,y:0,scale:1};
}
function resize(){const d=devicePixelRatio||1;canvas.width=canvas.clientWidth*d;canvas.height=canvas.clientHeight*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function screen(p){return{x:p.x*transform.scale+transform.x,y:p.y*transform.scale+transform.y}}
function draw(){
 ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);ctx.save();
 for(const edge of view.edges){const a=screen(view.positions.get(edge.subject_id)),b=screen(view.positions.get(edge.object_id));ctx.strokeStyle=colors[edge.relation]+"88";ctx.lineWidth=edge.relation==="prerequisite_of"?2:1;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
 for(const node of view.nodes){const p=screen(view.positions.get(node.id)),r=Math.max(4,Math.min(11,4+Math.sqrt(degree.get(node.id)||0)));ctx.fillStyle=typeColors[primaryType(node)]||"#a6da95";ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();
   if(node.id===root||degree.get(node.id)>=6){ctx.fillStyle="#e7eef9";ctx.font="12px system-ui";ctx.fillText(node.canonical_name,p.x+r+3,p.y+4)}
 }ctx.restore();
}
function nodeAt(x,y){let best=null,dist=18;for(const n of view.nodes){const p=screen(view.positions.get(n.id)),d=Math.hypot(x-p.x,y-p.y);if(d<dist){best=n;dist=d}}return best}
function edgeAt(x,y){let best=null,dist=7;for(const e of view.edges){const a=screen(view.positions.get(e.subject_id)),b=screen(view.positions.get(e.object_id));const dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((x-a.x)*dx+(y-a.y)*dy)/(dx*dx+dy*dy||1)));const d=Math.hypot(x-(a.x+t*dx),y-(a.y+t*dy));if(d<dist){best=e;dist=d}}return best}
function showNode(n){const ev=evidenceByTarget.get(`entity:${n.id}`)||[];document.getElementById("details").textContent=[
 n.canonical_name,`ID: ${n.id}  degree: ${degree.get(n.id)||0}`,`aliases: ${(n.aliases||[]).join(" / ")}`,`definition: ${n.definition}`,
 `type_profile: ${JSON.stringify(n.type_profile)}`,...ev.slice(0,6).map((e,i)=>`\\nEvidence ${i+1} · ${e.source.name} · ${e.location}\\n原文: ${e.source_text}\\n模型引文: ${e.model_quote}`)
 ].join("\\n")}
function showEdge(e){const ev=evidenceByTarget.get(`claim:${e.id}`)||[];document.getElementById("details").textContent=[
 `${e.subject}  --${e.relation}-->  ${e.object}`,`Claim ID: ${e.id}`,...ev.map((x,i)=>`\\nEvidence ${i+1} · ${x.source.name} · ${x.location}\\n原文: ${x.source_text}\\n裁判: ${x.validation.verdict} · ${x.validation.reason}`)
 ].join("\\n")}
const statusLabels={pending_endpoint:"待定端点",pending_judgment:"待裁判",supported_unmaterialized:"支持但未落实",blocked:"物化阻塞",insufficient:"证据不足",contradicts:"证据矛盾"};
function endpointText(endpoint){return endpoint.entity_id?`${endpoint.name} → ${endpoint.entity_name} (#${endpoint.entity_id})`:`${endpoint.name} → 未解析`}
function showObservation(item){document.getElementById("details").textContent=[
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
canvas.addEventListener("click",e=>{if(drag?.moved)return;const n=nodeAt(e.offsetX,e.offsetY);if(n){showNode(n);return}const edge=edgeAt(e.offsetX,e.offsetY);if(edge)showEdge(edge)});
canvas.addEventListener("mousedown",e=>drag={x:e.clientX,y:e.clientY,ox:transform.x,oy:transform.y,moved:false});
addEventListener("mousemove",e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved=Math.abs(dx)+Math.abs(dy)>3;transform.x=drag.ox+dx;transform.y=drag.oy+dy;draw()});
addEventListener("mouseup",()=>{drag=null});
canvas.addEventListener("wheel",e=>{e.preventDefault();const factor=e.deltaY<0?1.12:.89,old=transform.scale;transform.scale=Math.max(.3,Math.min(4,old*factor));transform.x=e.offsetX-(e.offsetX-transform.x)*transform.scale/old;transform.y=e.offsetY-(e.offsetY-transform.y)*transform.scale/old;draw()},{passive:false});
document.getElementById("search").addEventListener("keydown",e=>{if(e.key!=="Enter")return;const q=e.target.value.trim().toLowerCase();const hit=data.entities.find(x=>x.canonical_name.toLowerCase()===q)||data.entities.find(x=>x.canonical_name.toLowerCase().includes(q)||(x.aliases||[]).some(a=>a.toLowerCase().includes(q)));if(hit){root=hit.id;build();showNode(hit)}});
document.getElementById("reset").onclick=()=>{root=[...degree.entries()].sort((a,b)=>b[1]-a[1])[0][0];build()};
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
