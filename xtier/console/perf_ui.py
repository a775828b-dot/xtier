"""One-click performance workspace and timestamped monitoring charts."""
HTML = '<style>\n.xt .xp-layout{display:grid;grid-template-columns:280px minmax(0,1fr);gap:20px;align-items:start}.xt .xp-options{padding:16px;background:#080f20;border:1px solid var(--line);border-radius:12px}.xt .xp-options .xt-field{margin-bottom:12px}.xt .xp-options p,.xt .xp-caption{font-size:12px;color:var(--d)}.xt .xp-start{box-sizing:border-box;width:100%;justify-content:center;margin-top:12px}.xt .xp-toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}.xt .xp-range{display:flex;gap:4px}.xt .xp-range button[aria-pressed=true]{background:#14364a;border-color:#3ee6ff;color:#fff}.xt .xp-dashboards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.xt .xp-card{background:#080f20;border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}.xt .xp-gpu,.xt .xp-queue{grid-column:1/-1}.xt .xp-card.xp-gpu svg{height:150px}.xt .xp-gpu-values{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}.xt .xp-gpu-values span{font-size:12px}.xt .xp-card .xp-gpu-values strong{display:inline;font-size:18px;margin-left:6px}.xt .xp-card header{display:flex;justify-content:space-between;gap:8px;color:var(--d);font-size:12px}.xt .xp-card strong{display:block;font-size:25px;color:#e4f8ff;margin:6px 0}.xt .xp-card svg{width:100%;height:100px;display:block}.xt .xp-history-row{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 0;border-bottom:1px solid var(--line)}.xt .xp-history-row small{display:block;color:var(--d)}.xt .xp-error{color:#fb7185}.xt .xp-live-extra{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--d);margin:12px 0}.xt .xp-details{margin-top:16px}.xp-dialog{background:#080f20;color:#dcecff;border:1px solid #31566d;border-radius:16px;width:min(1100px,94vw);max-height:90vh;padding:0;font:14px/1.5 system-ui,sans-serif;box-sizing:border-box}.xp-dialog::backdrop{background:rgba(0,0,0,.75);backdrop-filter:blur(4px)}.xp-dialog .xp-dialog-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:18px 22px;border-bottom:1px solid #20364a;position:sticky;top:0;background:#080f20;z-index:1}.xp-dialog h3,.xp-dialog h4{margin:0 0 12px}.xp-dialog .xp-dialog-head h3{margin:0}.xp-dialog .xp-dialog-body{padding:22px}.xp-dialog button,.xp-dialog a{color:#b8f5ff;background:#10263b;border:1px solid #31566d;border-radius:8px;padding:7px 12px;cursor:pointer}.xp-dialog .xp-config{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin-bottom:22px}.xp-dialog .xp-config div{padding:10px;background:#101c30;border-radius:8px;overflow-wrap:anywhere}.xp-dialog dt{color:#92a8bf;font-size:12px}.xp-dialog dd{margin:3px 0 0}.xp-dialog table{border-collapse:collapse;min-width:760px;width:100%;font-size:12px}.xp-dialog th,.xp-dialog td{padding:10px;text-align:left;border-bottom:1px solid #20364a}.xp-dialog th{color:#8fdce8}.xp-dialog pre{white-space:pre-wrap;overflow-wrap:anywhere}.xp-dialog .xp-summary{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}.xp-dialog .xp-scroll{overflow-x:auto}.xt .xt-perf{margin:0 22px 18px}\n@container xt (max-width:850px){.xt .xp-layout{grid-template-columns:1fr}.xt .xp-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px}.xt .xp-options .xp-wide{grid-column:1/-1}}\n@container xt (max-width:480px){.xt .xp-dashboards,.xt .xp-options{grid-template-columns:1fr}.xt .xp-history-row{align-items:flex-start;flex-direction:column}.xt .xt-perf{margin:0 12px 14px}}\n</style>\n<div class="xt-panel xt-perf" id="xt-perf">\n<div class="xp-toolbar"><div><h4>性能实验室</h4><div class="xp-caption">选择测试，一键运行。自动保存本次在线服务配置和测试结果。</div></div><span id="xp-status" role="status" aria-live="polite">等待连接</span></div>\n<div class="xp-layout">\n<aside class="xp-options">\n<label class="xt-field">测试项目<select id="xp-kind"><option value="decode">固定输出 decode</option><option value="concurrency">并发阶梯 1 / 2 / 4 / 8 / 16</option><option value="context">上下文 1K / 8K / 32K / 64K</option><option value="cache">前缀缓存冷 / 热</option><option value="agent">Agent 64K 多轮任务</option><option value="copy">照抄正确性（2 段）</option></select></label>\n<label class="xt-field">每请求输出上限<select id="xp-tokens"><option>128</option><option>256</option><option selected>512</option><option>1024</option></select></label>\n<label class="xt-field">重复轮数<select id="xp-rounds"><option>1</option><option>2</option><option selected>3</option><option>5</option></select></label>\n<label class="xt-field">Agent 最大交互步数<select id="xp-steps"><option>4</option><option selected>8</option><option>16</option><option>24</option></select></label>\n<label class="xp-wide xp-caption"><input type="checkbox" id="xp-queue"> 包含超过服务上限的排队压力测试</label>\n<p class="xp-wide" id="xp-help"></p>\n<div class="xp-wide"><button class="xt-btn xp-start" id="xp-run">开始测试</button><button class="xt-btn xp-start" id="xp-stop" disabled>停止测试</button><button class="xt-btn xp-start" id="xp-view" disabled>查看测试结果</button></div>\n<p class="xp-wide">测试当前在线配置，不应用上方尚未生效的修改。测试期间请暂停其他客户端请求。</p>\n<div class="xp-wide xp-error" id="xp-error" role="alert"></div>\n</aside>\n<div><div class="xp-toolbar"><b>实时性能</b><div class="xp-range" role="group" aria-label="监控时间范围"><button class="xt-btn" data-window="60" aria-pressed="true">最近 1 分钟</button><button class="xt-btn" data-window="300" aria-pressed="false">最近 5 分钟</button></div></div>\n<div class="xp-dashboards" id="xp-dashboards"></div><div class="xp-live-extra" id="xp-live-extra"></div><p class="xp-caption" id="xp-chart-note">曲线从打开页面开始采样；横轴为真实时间，空白表示尚无数据。</p>\n<details class="xp-details"><summary>调度阶段与指标口径</summary><pre id="xp-stages" style="white-space:pre-wrap"></pre><p class="xp-caption">CPU 为最近日志窗口，可能早于本次请求。调度阶段允许与 GPU 执行重叠，不能简单相加。纯 GPU 内核耗时与 Engram 独立等待未采集。TTFT 包含排队、prefill 和传输。</p></details>\n</div></div>\n<details class="xp-details"><summary>按指标调优</summary><ul id="xp-advice"></ul></details>\n<div class="xp-toolbar" style="margin-top:24px"><h4>测试记录</h4><span class="xp-caption">配置在开始时保存，后续改参数不影响历史记录</span></div><div id="xp-history">暂无测试记录</div>\n<details class="xp-details"><summary>当前测试日志</summary><pre id="xp-log" style="max-height:220px;overflow:auto;white-space:pre-wrap"></pre></details>\n<dialog class="xp-dialog" id="xp-dialog" aria-labelledby="xp-dialog-title"><div class="xp-dialog-head"><h3 id="xp-dialog-title">测试结果</h3><button id="xp-close" aria-label="关闭测试结果">关闭</button></div><div class="xp-dialog-body" id="xp-dialog-body"></div></dialog>\n</div>\n'
JS = r'''
(() => {
 const el=id=>document.getElementById(id); if(!el('xt-perf'))return;
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const fmt=n=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(2):'—';
 const states={idle:'准备就绪',running:'测试运行中',cancelling:'正在停止',cancelled:'已停止',done:'测试完成',failed:'测试失败',interrupted:'测试已中断'};
 const kinds={decode:'固定输出 decode',concurrency:'并发阶梯',context:'上下文长度',cache:'前缀缓存冷/热',agent:'Agent 多轮',copy:'照抄正确性'};
 const labels={packs:'加速包',spec:'投机解码',sps:'SPS 校准',context:'上下文容量',threads:'CPU 专家线程',slots:'基础热专家槽',elastic:'弹性热专家',engram:'Engram 缓存 GiB',cold:'冷专家卸载',mem:'显存静态比例',tokens:'输出上限',rounds:'重复轮数',steps:'Agent 最大步数',queue:'排队压力',kind:'测试项目'};
 let busy=false, actionBusy=false, history=[], latestId=null, dialogId=null, historySignature='', windowSeconds=60, points=[], prev=null, timer;
 const help={decode:'固定短提示、固定输出上限、独立前缀。测单请求生成和首 token 延迟。',concurrency:'每档实际同时提交对应请求数。默认跳过超过服务上限的档位；排队模式用于吞吐与排队压力，不能视作对应规模的 GPU batch。',context:'使用本地 tokenizer 构造目标长度，实际 prompt token 以返回 usage 为准；超出上下文余量自动跳过。',cache:'相同前缀连续请求两次，观察 cached_tokens 与 TTFT；不清空全局缓存，唯一前缀作为近似冷启动。',agent:'复用 bench/agent_bench.py：64K 编码代理、多轮工具调用。只修改临时沙箱，不执行模型生成代码。耗时较长，输出不固定；以任务结果、接受长度、验证步速综合判断。',copy:'复用 bench/copy_test.py：逐字照抄 2 段源码，报告差异。输出上限、轮数和 Agent 步数不适用于此测试。'};
 const definitions=[['throughput','输出吞吐','tok/s','#3ee6ff'],['prefill','Prefill 吞吐','tok/s','#38bdf8'],['hit','显存专家命中 · 最近日志窗口','%','#f472b6'],['cpu','CPU 每层调用 · 最近日志窗口','µs','#9cf06b'],['queue','排队请求','个','#fb7185']];
 const gpuSeries=[['power','功耗','W','#fbbf24'],['util','利用率','%','#a78bfa'],['memory','显存','GiB','#f472b6']];
 const axes=(key,width=360)=>`<line x1="32" x2="${width-8}" y1="76" y2="76" stroke="#20364a"/><line x1="32" x2="${width-8}" y1="14" y2="14" stroke="#20364a" stroke-dasharray="3 4"/><text id="xp-max-${key}" x="2" y="12" fill="#7f90ad" font-size="10"></text><text id="xp-from-${key}" x="32" y="96" fill="#7f90ad" font-size="10"></text><text x="${width-34}" y="96" fill="#7f90ad" font-size="10">现在</text>`;
 el('xp-dashboards').innerHTML=`<section class="xp-card xp-gpu"><header><span>GPU 功耗 / 利用率 / 显存</span><span>相对上限 %</span></header><div class="xp-gpu-values">${gpuSeries.map(([k,t,u,c])=>`<span style="color:${c}">${t}<strong id="xp-value-${k}">—</strong> ${u}</span>`).join('')}</div><svg viewBox="0 0 720 100" role="img" aria-label="GPU 功耗、利用率与显存变化曲线">${axes('gpu',720)}${gpuSeries.map(([k,t,u,c])=>`<path id="xp-line-${k}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke"/>`).join('')}</svg><p class="xp-caption">功耗 / 功耗上限、利用率、显存 / 总显存采用相对刻度；上方显示实际数值。</p></section>`+definitions.map(([key,title,unit,color])=>`<section class="xp-card ${key==='queue'?'xp-queue':''}"><header><span>${title}</span><span>${unit}</span></header><strong id="xp-value-${key}">—</strong><svg viewBox="0 0 360 100" role="img" aria-label="${title}变化曲线">${axes(key)}<path id="xp-line-${key}" fill="none" stroke="${color}" stroke-width="2"/></svg></section>`).join('');
 function line(visible,key,start,max,width=360){
  let path='',last=null;
  visible.forEach(p=>{if(!Number.isFinite(p[key])){last=null;return;}const x=32+(p.time-start)/windowSeconds*(width-40),y=76-Math.max(0,p[key])/max*62;path+=(last&&p.time-last.time<=20?'L':'M')+x.toFixed(1)+','+y.toFixed(1)+' ';last=p;});
  return path;
 }
 function chart(){
  const now=Date.now()/1000, start=now-windowSeconds, visible=points.filter(p=>p.time>=start&&p.time<=now);
  definitions.forEach(([key])=>{
   const valid=visible.filter(p=>Number.isFinite(p[key]));const max=key==='hit'?100:Math.max(1,...valid.map(p=>p[key]))*1.1;
   el('xp-line-'+key).setAttribute('d',line(visible,key,start,max));el('xp-max-'+key).textContent=fmt(max);el('xp-from-'+key).textContent=windowSeconds===60?'−1 分钟':'−5 分钟';
  });
  const max=Math.max(100,...visible.flatMap(p=>gpuSeries.map(([k])=>p['gpu_'+k]).filter(Number.isFinite)));
  gpuSeries.forEach(([k])=>el('xp-line-'+k).setAttribute('d',line(visible,'gpu_'+k,start,max,720)));
  el('xp-max-gpu').textContent=fmt(max)+'%';el('xp-from-gpu').textContent=windowSeconds===60?'−1 分钟':'−5 分钟';
 }
 document.querySelectorAll('[data-window]').forEach(b=>b.onclick=()=>{windowSeconds=Number(b.dataset.window);document.querySelectorAll('[data-window]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));chart();});
 function draw(live){
  const g=(live.gpus||[])[0]||{},m=live.metrics||{},c=live.cpu_window;
  const sameService=live.service_id===prev?.service_id;
  const rate=(name,key)=>{const current=m[name],prior=prev?.metrics?.[name];return sameService?(live.time===prev?.time?points.at(-1)?.[key]:(live.time>prev?.time&&Number.isFinite(current)&&Number.isFinite(prior)&&current>=prior?(current-prior)/(live.time-prev.time):null)):null;};
  const throughput=rate('sglang:generation_tokens_total','throughput'),prefill=rate('sglang:prefill_effective_tokens_total','prefill');
  const point={time:live.time,throughput,prefill,hit:c?.hit==null?null:c.hit*100,power:g.power_w,util:g.util_pct,memory:g.memory_mib==null?null:g.memory_mib/1024,cpu:c?.cpu_us,queue:m['sglang:num_queue_reqs'],gpu_power:Number.isFinite(g.power_w)&&g.power_limit_w>0?g.power_w/g.power_limit_w*100:null,gpu_util:g.util_pct,gpu_memory:Number.isFinite(g.memory_mib)&&g.memory_total_mib>0?g.memory_mib/g.memory_total_mib*100:null};
  if(live.time&&live.time!==prev?.time){
   points.push(point);points=points.filter(p=>p.time>=live.time-300);
   const stages=[];
   if(sameService&&prev&&live.time>prev.time)for(const[k,v]of Object.entries(m))if(k.startsWith('sglang:scheduler_stage_seconds_total')&&prev.metrics[k]!=null&&v>=prev.metrics[k]){const category=k.match(/category="([^"]+)"/)?.[1]||'other';const names={other:'其他调度工作',recv_requests:'接收请求',process_input_requests:'处理输入',get_next_batch_to_run:'组建批次',process_queue:'处理队列',run_batch:'提交/运行批次（非纯 GPU 时间）',process_batch_result:'处理结果',sanity_check_cache:'缓存检查',idle:'空闲'};stages.push((names[category]||category)+': '+fmt((v-prev.metrics[k])/(live.time-prev.time)*100)+'% 观测墙钟');}
   el('xp-stages').textContent=stages.join('\n')||'等待两个有效采样点';prev=live;
  }
  [...definitions,...gpuSeries].forEach(([k])=>el('xp-value-'+k).textContent=fmt(point[k]));
  el('xp-live-extra').textContent=`运行请求 ${fmt(m['sglang:num_running_reqs'])} · 专家命中 ${c?fmt(c.hit*100)+'%':'—'} · 接受长度 ${fmt(m['sglang:spec_accept_length'])} · SM / 显存时钟 ${fmt(g.sm_mhz)} / ${fmt(g.mem_mhz)} MHz`;
  const note=live.service_state==='online'?'曲线自打开页面开始累积，切换时间范围不重启测试。输出和 prefill 吞吐为服务全局有效 token 计数增量 / 采样墙钟，非纯内核处理速率；CPU 与显存专家命中为最近日志窗口，可能是旧数据。':'服务'+({loading:'加载中',offline:'未启动',failed:'已退出'}[live.service_state]||'状态待确认')+'；硬件监控持续采样，推理指标待服务在线后提供。';
  el('xp-chart-note').textContent=note+((live.errors||[]).length?' 采样异常：'+live.errors.join('；'):'');chart();
 }
 const form=()=>{const data=new URLSearchParams({kind:el('xp-kind').value});for(const key of ['tokens','rounds','steps','queue']){const control=el('xp-'+key);if(!control.disabled)data.set(key,key==='queue'?(control.checked?'1':'0'):control.value);}return data;};
 el('xp-kind').onchange=()=>{const k=el('xp-kind').value;el('xp-help').textContent=help[k];el('xp-steps').disabled=k!=='agent';el('xp-rounds').disabled=['agent','copy','cache'].includes(k);el('xp-tokens').disabled=k==='copy';el('xp-queue').disabled=k!=='concurrency';};el('xp-kind').onchange();
 async function action(name){if(actionBusy)return;actionBusy=true;el('xp-run').disabled=true;el('xp-error').textContent='';try{const r=await fetch('/api/performance/'+name,{method:'POST',headers:{'X-Requested-With':'fetch'},body:form()});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);if(d.id){latestId=d.id;el('xp-view').disabled=false;}el('xp-status').textContent=name==='start'?'测试已启动，配置已保存':'正在停止测试';}catch(e){el('xp-error').textContent=e.message;}finally{actionBusy=false;poll();}}
 el('xp-run').onclick=()=>action('start');el('xp-stop').onclick=()=>action('cancel');el('xp-view').onclick=()=>showReport(latestId);
 el('xp-close').onclick=()=>el('xp-dialog').close();el('xp-dialog').addEventListener('close',()=>{dialogId=null;});
 function configGrid(config){return '<dl class="xp-config">'+Object.entries(config||{}).map(([k,v])=>'<div><dt>'+esc(labels[k]||k)+'</dt><dd>'+esc(k==='kind'?(kinds[v]||v):typeof v==='object'?JSON.stringify(v):v)+'</dd></div>').join('')+'</dl>';}
 function resultTable(headers, rows){return '<div class="xp-scroll"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(cell=>'<td>'+esc(cell)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
 function agentResult(agent){
  const rows=agent.rows||[], total=rows.reduce((n,r)=>n+(r.output_tokens||0),0);
  const failures=agent.args?.failures||[];
  let html='<h4>Agent 多轮结果</h4>'+configGrid({'完成请求':rows.length,'输出 token':total,'总墙钟秒':fmt(agent.wall_s),'请求失败':failures.length,'任务准确率':'未评分：此测试没有标准答案或任务验收器'});
  html+='<p>下表逐轮展示性能和工具调用。工具调用次数、无 HTTP 错误均不代表任务正确。</p>';
  html+=resultTable(['会话 / 步骤','输入 / 缓存 / 输出 token','TTFT 秒','decode tok/s','验证步数','接受长度','每验证步 ms','工具调用'],rows.map(r=>[
   r.session+' / '+r.step,[r.prompt_tokens??'—',r.cached_tokens??'—',r.output_tokens??'—'].join(' / '),fmt(r.ttft_s),fmt(r.decode_tps),r.verify_steps||'—',r.verify_steps?fmt(r.accept):'—',r.verify_steps?fmt(r.step_ms):'—',(r.tool_calls||[]).join(', ')||'无'
  ]));
  if(failures.length) html+='<h4>未完成请求</h4>'+resultTable(['会话','步骤','HTTP 状态'],failures.map(r=>[r.session,r.step,r.http_status]));
  if(!rows.length) html+='<p>没有完成的请求，不能据此评估性能。</p>';
  return html;
 }
 function copyResult(result){
  const rows=result.rows||[], total=result.total||0, correct=result.correct??(total-(result.different||0));
  const accuracy=result.accuracy??(total?correct/total:null);
  let html='<h4>照抄正确性</h4>'+configGrid({'归一化完全匹配准确率':accuracy==null?'—':fmt(accuracy*100)+'%','严格原文匹配率':result.strict_accuracy==null?'—':fmt(result.strict_accuracy*100)+'%','通过 / 总例数':correct+' / '+total,'请求错误':result.errors??'未记录'});
  html+='<p>准确率 = 完全匹配的例数 / 总例数。归一化沿用照抄脚本：去除首尾换行及外层代码围栏；严格匹配不做归一化。请求错误计为未通过。两段样本只能作为冒烟检查。</p>';
  html+=resultTable(['样例','结果','差异片段数','结束原因','请求错误'],rows.map((r,i)=>[r.case||i+1,r.exact?'完全匹配':'未通过',r.diffs.length,r.finish_reason||'未记录',r.error||'无']));
  rows.forEach((r,i)=>{html+='<details><summary>样例 '+(r.case||i+1)+' · 查看原文、输出与差异位置</summary><h4>原文</h4><pre>'+esc(r.src)+'</pre><h4>模型输出</h4><pre>'+esc(r.out)+'</pre>'+resultTable(['操作','原文字符范围 [起,止)','输出字符范围 [起,止)'],r.diffs.map(d=>[d[0],d[1]+'–'+d[2],d[3]+'–'+d[4]]))+'<p>差异位置以归一化后的字符为准。</p></details>';});
  if(!rows.length) html+='<p>该历史结果没有逐例数据。</p>';
  return html;
 }
 async function showReport(id){
  if(!id)return;dialogId=id;if(!el('xp-dialog').open)el('xp-dialog').showModal();el('xp-dialog-body').textContent='正在读取保存的配置和结果…';
  try{const response=await fetch('/api/performance/report?id='+encodeURIComponent(id),{signal:AbortSignal.timeout(15000)});const r=await response.json();if(!response.ok)throw Error(r.error||response.statusText);if(dialogId!==id)return;
   const snap=r.launch_snapshot||{},settings=snap.test_options||r.settings||{}, config=snap.configuration||r.configuration||{};
   const test=Object.fromEntries(['kind','tokens','rounds','steps','queue'].filter(k=>settings[k]!=null).map(k=>[k,settings[k]]));
   let body='<p>'+esc(id)+' · '+esc(states[snap.state]||snap.state||'历史结果')+' · '+esc(snap.model||r.settings?.model||'')+'</p><h4>本次运行配置</h4>'+configGrid(config)+'<h4>测试设置</h4>'+configGrid(test)+'<h4>测试结果</h4>';
   if(r.pending)body+='<p>'+esc(r.note||'测试仍在运行，结果完成后可查看；本次配置已保存。')+'</p>';
   if(r.error)body+='<p style="color:#fb7185">'+esc(r.error)+'</p>';
   if(r.rounds?.length){body+='<div class="xp-scroll"><table><thead><tr><th>场景 / 轮次</th><th>总 tok/s</th><th>单请求 decode</th><th>TTFT P50 / P95 秒</th><th>完整延迟 P95 秒</th><th>实际输入 / 缓存 / 输出 token</th><th>错误 / 备注</th></tr></thead><tbody>';
    for(const row of r.rounds)body+='<tr><td>'+esc('并发 '+(row.concurrency||1)+' · '+(row.context_target||row.context||'短')+' · '+({cold:'冷前缀',warm:'热前缀',independent:'独立请求'}[row.phase]||'')+' #'+((row.repeat??0)+1))+'</td><td>'+fmt(row.aggregate_tps)+'</td><td>'+fmt(row.decode_tps_median)+'</td><td>'+fmt(row.ttft_p50_s)+' / '+fmt(row.ttft_p95_s)+'</td><td>'+fmt(row.latency_p95_s)+'</td><td>'+esc((row.requests||[]).map(q=>[q.prompt_tokens??'—',q.cached_tokens??'—',q.output_tokens??'—'].join('/')).join(', '))+'</td><td>'+esc(row.skipped||(row.queued?'含排队；':'')+(row.errors??0))+'</td></tr>';
    body+='</tbody></table></div>';}
   if(r.observations)body+='<p>采样平均功耗 '+fmt(r.observations.sampled_gpu_mean_w)+' W · 采样能量 '+fmt(r.observations.sampled_gpu_energy_j)+' J · 覆盖 '+fmt(r.observations.sampled_duration_s)+' 秒（非完整请求能耗）</p>';
   if(r.dspark){body+='<h4>DSpark / SPS 实际运行证据</h4>';if(r.dspark.error)body+='<p>'+esc(r.dspark.error)+'</p>';else{body+=configGrid({'成本表':r.dspark.table||'未配置'});body+=resultTable(['rank','验证模式','观测步数','有成本预测的预算决策','缩短预算步数','预算分布','模拟接受'],(r.dspark.ranks||[]).map(x=>[x.rank,x.mode,x.observed_steps,x.planned_steps,x.reduced_steps,Object.entries(x.budget_counts).map(([k,v])=>k+': '+v+'步').join(', '),x.simulated_acceptance??'关闭']));if(!r.dspark.ranks?.length)body+='<p>未采集运行记录，不能仅凭表路径认定 SPS 生效。</p>';body+='<p>'+esc(r.dspark.note)+'</p>';}}
   if(r.agent)body+=agentResult(r.agent);
   if(r.copy_results)body+=copyResult(r.copy_results);
   body+='<p>小样本 P95 仅作参考。CPU 窗口和调度阶段可能重叠；纯 GPU / Engram 独立耗时未采集。</p><a href="/api/performance/report?id='+encodeURIComponent(id)+'" download="'+esc(id)+'.json">导出完整结果 JSON</a>';
   el('xp-dialog-body').innerHTML=body;
  }catch(e){if(dialogId===id)el('xp-dialog-body').textContent='读取结果失败：'+e.message;}
 }
 async function poll(){clearTimeout(timer);if(busy)return;busy=true;try{
  const response=await fetch('/api/performance',{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error(response.statusText);const d=await response.json();draw(d.live);
  el('xp-run').disabled=actionBusy||['running','cancelling'].includes(d.job.state)||d.ready!==true;el('xp-run').title=d.ready===true?'':d.blocked_reason||'服务尚未就绪，暂不能开始测试';el('xp-stop').disabled=d.job.state!=='running';el('xp-status').textContent=states[d.job.state]||d.job.state;
  el('xp-log').textContent=(d.job.tail||[]).join('\n');el('xp-advice').innerHTML=d.advice.map(s=>'<li>'+esc(s)+'</li>').join('');
  if(d.job.id)latestId=d.job.id;else if(!latestId&&d.history.length)latestId=d.history[0].id;el('xp-view').disabled=!latestId;
  const signature=JSON.stringify(d.history.map(h=>[h.id,h.state]));
  if(signature!==historySignature){historySignature=signature;history=d.history;
   el('xp-history').innerHTML=history.map(h=>'<div class="xp-history-row"><div><b>'+esc(kinds[h.test_options?.kind]||'性能测试')+'</b><small>'+esc(h.id+' · '+(states[h.state]||h.state))+'</small></div><button class="xt-btn" data-report="'+esc(h.id)+'">查看测试结果</button></div>').join('')||'暂无测试记录，开始测试后会自动保存。';
   el('xp-history').querySelectorAll('[data-report]').forEach(b=>b.onclick=()=>showReport(b.dataset.report));if(dialogId)showReport(dialogId);
  }
 }catch(e){el('xp-run').disabled=true;el('xp-run').title='无法确认服务状态，暂不能开始测试';el('xp-status').textContent='监控连接失败';el('xp-error').textContent=e.message;}finally{busy=false;timer=setTimeout(poll,document.hidden?15000:3000);}}
 poll();
})();
'''
