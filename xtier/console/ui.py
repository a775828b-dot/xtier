"""The console page: markup, style and script (core.Console holds the logic).

A model rail (small cards: name, port, README decode / prefill speeds; it scrolls when the models outgrow it, so the
console keeps its height), the launch options of the chosen model (defaults = the measured configuration, changed
ones are marked) and a HUD that follows the loading phases and, once online, the server's decode telemetry. Always
dark, scoped under .xt, so a host page can embed card_html() with its own texts and routes (data['texts'],
data['api']); page_html() is the standalone page of `python -m xtier.console`.
"""

import html
import json

SUBTITLE = '官方 SGLang main + XTier 插件 · 专家分布在显存 / 内存 / NVMe 三层 · 每个模型在自己的端口上提供 OpenAI 兼容接口'
FOOT = '默认参数 = configs/*/console.py 里的实测配置，改动过的项会标出。一次只跑一个模型：启动新的模型前会先停掉当前的。'

CSS = r"""
.xt{--c:#3ee6ff;--v:#a78bfa;--p:#f472b6;--g:#9cf06b;--a:#fbbf24;--r:#fb7185;--t:#dcecff;--d:#7f90ad;--line:rgba(62,230,255,.22);
position:relative;isolation:isolate;container:xt/inline-size;margin:0 0 18px;border-radius:18px;overflow:hidden;color:var(--t);text-align:left;
background:radial-gradient(900px 340px at 8% -20%,rgba(62,230,255,.20),transparent 62%),radial-gradient(760px 320px at 105% -10%,rgba(167,139,250,.22),transparent 60%),radial-gradient(600px 300px at 50% 130%,rgba(244,114,182,.10),transparent 60%),linear-gradient(180deg,#070b19,#03050c 70%);
border:1px solid rgba(62,230,255,.38);box-shadow:inset 0 0 0 1px rgba(62,230,255,.07),0 24px 60px -28px rgba(62,230,255,.55);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.xt::before{content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;
background-image:linear-gradient(rgba(62,230,255,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(62,230,255,.055) 1px,transparent 1px);
background-size:26px 26px;-webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 85%);mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 85%)}
.xt::after{content:"";position:absolute;left:0;right:0;top:-40%;height:40%;z-index:-1;pointer-events:none;
background:linear-gradient(180deg,transparent,rgba(62,230,255,.07) 70%,rgba(62,230,255,.16) 92%,transparent);animation:xt-scan 7s linear infinite}
@keyframes xt-scan{to{transform:translateY(350%)}}
.xt *{box-sizing:border-box}
.xt-mono,.xt code,.xt-num{font-family:ui-monospace,"SFMono-Regular","JetBrains Mono",Menlo,Consolas,monospace}
.xt-head{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:18px 22px 12px;border-bottom:1px solid var(--line)}
.xt-brand{display:flex;align-items:center;gap:14px;min-width:0}
.xt-logo{width:46px;height:46px;flex:none;filter:drop-shadow(0 0 10px rgba(62,230,255,.6))}
.xt-logo .spin{transform-origin:24px 24px;animation:xt-rot 14s linear infinite}
@keyframes xt-rot{to{transform:rotate(360deg)}}
.xt-title{font-weight:800;letter-spacing:.22em;font-size:19px;line-height:1.1;background:linear-gradient(90deg,#e9fbff,var(--c) 45%,var(--v));-webkit-background-clip:text;background-clip:text;color:transparent}
.xt-title span{display:inline-block;margin-left:10px;font-size:11px;letter-spacing:.32em;color:var(--d);-webkit-text-fill-color:var(--d)}
.xt-sub{margin-top:4px;color:var(--d);font-size:12.5px}
.xt-state{display:flex;align-items:center;gap:10px;padding:8px 14px;border:1px solid var(--line);border-radius:999px;background:rgba(4,10,24,.6);white-space:nowrap;min-width:0}
.xt-state b{letter-spacing:.2em;font-size:13px}
.xt-state small{color:var(--d);overflow:hidden;text-overflow:ellipsis}
.xt-led{flex:none;width:10px;height:10px;border-radius:50%;background:#45506a;box-shadow:0 0 0 3px rgba(69,80,106,.25)}
.xt[data-state=online] .xt-led{background:var(--g);box-shadow:0 0 12px var(--g),0 0 0 3px rgba(156,240,107,.2);animation:xt-pulse 1.8s ease-in-out infinite}
.xt[data-state=loading] .xt-led{background:var(--a);box-shadow:0 0 12px var(--a);animation:xt-blink .9s steps(2) infinite}
.xt[data-state=failed] .xt-led{background:var(--r);box-shadow:0 0 12px var(--r)}
.xt[data-state=online] .xt-state b{color:var(--g)}.xt[data-state=loading] .xt-state b{color:var(--a)}.xt[data-state=failed] .xt-state b{color:var(--r)}
@keyframes xt-pulse{50%{box-shadow:0 0 22px var(--g),0 0 0 6px rgba(156,240,107,.08)}}
@keyframes xt-blink{50%{opacity:.35}}
.xt-body{display:grid;grid-template-columns:minmax(220px,264px) minmax(0,1fr) minmax(0,1fr);gap:14px;padding:16px 22px 20px}
.xt-panel{position:relative;min-width:0;padding:14px 16px;border:1px solid var(--line);border-radius:14px;background:rgba(5,11,26,.72);backdrop-filter:blur(3px)}
.xt-panel h4{margin:0 0 10px;font-size:11px;line-height:1.5;letter-spacing:.28em;color:var(--c);font-weight:700}
.xt-panel h4::before{content:"▌";margin-right:6px;color:var(--v)}
.xt-rail{padding:14px 12px 12px}
.xt-legend{margin:-6px 0 9px;font-size:10px;letter-spacing:.06em;color:var(--d)}
.xt-models{display:grid;grid-template-columns:minmax(0,1fr);gap:8px;align-content:start;max-height:404px;overflow-y:auto;margin-right:-6px;padding:1px 6px 1px 0;scrollbar-width:thin;scrollbar-color:rgba(62,230,255,.35) transparent}
.xt-model{all:unset;cursor:pointer;position:relative;display:block;padding:1px;background:linear-gradient(135deg,rgba(62,230,255,.32),rgba(167,139,250,.16) 50%,rgba(62,230,255,.07));
clip-path:polygon(9px 0,100% 0,100% calc(100% - 9px),calc(100% - 9px) 100%,0 100%,0 9px);transition:filter .18s}
.xt-model>div{padding:7px 11px 8px;background:linear-gradient(180deg,rgba(9,17,36,.96),rgba(5,9,20,.96));clip-path:inherit}
.xt-model:hover{filter:brightness(1.18)}
.xt-model:focus-visible>div{box-shadow:inset 0 0 0 2px var(--c)}
.xt-model[aria-selected=true]{background:linear-gradient(135deg,var(--c),var(--v) 55%,var(--p))}
.xt-model[aria-selected=true]>div{background:linear-gradient(180deg,rgba(14,30,58,.97),rgba(6,12,28,.97))}
.xt-model .top{display:flex;align-items:center;justify-content:space-between;gap:6px;min-width:0}
.xt-model .code{min-width:0;font-size:9.5px;letter-spacing:.18em;color:var(--c);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xt-model .name{margin-top:1px;font-weight:700;font-size:13.5px;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xt-model .perf{display:grid;grid-template-columns:32px minmax(0,1fr) auto;margin-top:3px;font-size:11px;line-height:1.45}
.xt-model .perf span{font-size:9.5px;letter-spacing:.12em;color:var(--d);align-self:center}
.xt-model .perf b{font-weight:600;color:#e3f3ff;white-space:nowrap}
.xt-model .perf .kind{grid-column:3;grid-row:1/span 2;align-self:center;font-style:normal;font-size:8.5px;line-height:1.5;letter-spacing:.12em;padding:0 5px;border-radius:5px;border:1px solid rgba(127,144,173,.4);color:var(--d)}
.xt-model .perf .kind[data-kind=agent]{color:var(--g);border-color:rgba(156,240,107,.5)}
.xt-model .run{flex:none;display:none;font-size:8.5px;line-height:1.5;letter-spacing:.16em;padding:0 6px;border-radius:999px;color:#04110a;background:var(--g);box-shadow:0 0 10px rgba(156,240,107,.7)}
.xt-model[data-running=loading] .run{display:block;background:var(--a);box-shadow:0 0 10px rgba(251,191,36,.6)}
.xt-model[data-running=online] .run{display:block}
.xt-sel{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 10px;margin:-2px 0 12px;padding-bottom:10px;border-bottom:1px dashed rgba(62,230,255,.16)}
.xt-sel b{font-size:15px;color:#f4fbff}
.xt-sel span{font-size:12px;color:var(--d)}
.xt-sel .hw{font-family:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;font-size:11px;letter-spacing:.06em;color:var(--c)}
.xt-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 12px}
.xt-field{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--d)}
.xt-field select{appearance:none;-webkit-appearance:none;width:100%;margin:0;padding:8px 30px 8px 10px;color:var(--t);font-family:inherit;font-size:13px;line-height:1.3;border-radius:9px;border:1px solid rgba(62,230,255,.28);
background:rgba(3,8,20,.9) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' fill='none' stroke='%233ee6ff' stroke-width='1.6'/%3E%3C/svg%3E") no-repeat right 10px center;transition:border-color .15s,box-shadow .15s}
.xt-field select:hover{border-color:rgba(62,230,255,.55)}
.xt-field select:focus{outline:none;border-color:var(--c);box-shadow:0 0 0 3px rgba(62,230,255,.18)}
.xt-field select:disabled{opacity:.42}
.xt-field.changed select{border-color:var(--v);box-shadow:0 0 0 1px rgba(167,139,250,.35)}
.xt-field.changed::after{content:"已改动 · 非实测默认";font-size:10.5px;color:var(--v)}
.xt-notes{margin-top:10px;min-height:18px;font-size:12px;color:var(--d)}
.xt-notes b{color:var(--a);font-weight:600}
.xt-actions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:14px}
.xt-btn{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:8px;padding:10px 16px;font-size:13px;font-weight:700;letter-spacing:.08em;border:1px solid var(--line);border-radius:10px;color:var(--t);background:rgba(62,230,255,.05);transition:all .15s}
.xt-btn:hover{border-color:var(--c);color:#fff;background:rgba(62,230,255,.12)}
.xt-btn:focus-visible{outline:2px solid var(--c);outline-offset:2px}
.xt-btn[disabled]{opacity:.38;pointer-events:none}
.xt-engage{position:relative;overflow:hidden;padding:12px 26px;font-size:15px;letter-spacing:.18em;color:#021018;border:0;border-radius:0;
background:linear-gradient(100deg,var(--c),#7ad8ff 40%,var(--v));clip-path:polygon(12px 0,100% 0,100% calc(100% - 12px),calc(100% - 12px) 100%,0 100%,0 12px);
box-shadow:0 0 24px rgba(62,230,255,.45);text-shadow:0 1px 0 rgba(255,255,255,.35)}
.xt-engage::after{content:"";position:absolute;top:0;bottom:0;width:40%;left:-60%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.65),transparent);transform:skewX(-20deg);animation:xt-shine 3.2s ease-in-out infinite}
@keyframes xt-shine{60%,100%{left:130%}}
.xt-engage:hover{color:#000;background:linear-gradient(100deg,#7ff3ff,#a9e6ff 40%,#c4b5fd);box-shadow:0 0 34px rgba(62,230,255,.7)}
.xt-abort{color:var(--r);border-color:rgba(251,113,133,.45);background:rgba(251,113,133,.06)}
.xt-abort:hover{color:#fff;border-color:var(--r);background:rgba(251,113,133,.18)}
.xt-hud{display:grid;grid-template-columns:150px minmax(0,1fr);gap:14px;align-items:center}
.xt-hud>div{min-width:0}
.xt-ring{position:relative;width:150px;height:150px}
.xt-ring svg{width:150px;height:150px;transform:rotate(-90deg)}
.xt-ring .track{fill:none;stroke:rgba(62,230,255,.12);stroke-width:8}
.xt-ring .ticks{fill:none;stroke:rgba(62,230,255,.35);stroke-width:2;stroke-dasharray:1 7.6}
.xt-ring .bar{fill:none;stroke:url(#xt-grad);stroke-width:8;stroke-linecap:round;transition:stroke-dashoffset .8s ease;filter:drop-shadow(0 0 6px rgba(62,230,255,.7))}
.xt-ring .core{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}
.xt-ring .core b{font-size:30px;line-height:1;color:#fff;text-shadow:0 0 14px rgba(62,230,255,.6)}
.xt-ring .core span{margin-top:5px;font-size:10.5px;letter-spacing:.2em;color:var(--d)}
.xt[data-state=online] .xt-ring .bar{stroke:var(--g);filter:drop-shadow(0 0 7px rgba(156,240,107,.75))}
.xt-phases{display:flex;flex-wrap:wrap;gap:5px;margin:0 0 10px;padding:0;list-style:none}
.xt-phases li{margin:0;font-size:10.5px;letter-spacing:.04em;padding:2px 6px;border-radius:6px;border:1px solid rgba(127,144,173,.25);color:#53607a;white-space:nowrap}
.xt-phases li.done{color:var(--c);border-color:rgba(62,230,255,.35)}
.xt-phases li.now{color:#03121a;background:var(--a);border-color:var(--a);box-shadow:0 0 12px rgba(251,191,36,.55)}
.xt[data-state=online] .xt-phases li.now{background:var(--g);border-color:var(--g);box-shadow:0 0 12px rgba(156,240,107,.6)}
.xt-meters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.xt-meter{padding:7px 9px;border:1px solid var(--line);border-radius:9px;background:rgba(62,230,255,.035)}
.xt-meter span{display:block;font-size:10.5px;letter-spacing:.12em;color:var(--d)}
.xt-meter b{font-size:17px;color:#f2fbff}
.xt-meter b small{font-size:11px;color:var(--d);font-weight:500;margin-left:2px}
.xt-spark{margin-top:10px;height:46px;border:1px solid var(--line);border-radius:9px;background:linear-gradient(180deg,rgba(62,230,255,.06),transparent);overflow:hidden}
.xt-spark svg{width:100%;height:100%;display:block}
.xt-spark .ln{fill:none;stroke:var(--c);stroke-width:1.6;filter:drop-shadow(0 0 3px rgba(62,230,255,.8))}
.xt-spark .ar{fill:rgba(62,230,255,.12);stroke:none}
.xt-gpu{margin-top:10px;font-size:11.5px;color:var(--d)}
.xt-gpu .bar{height:6px;margin-top:4px;border-radius:999px;background:rgba(62,230,255,.1);overflow:hidden}
.xt-gpu .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--c),var(--v),var(--p));box-shadow:0 0 10px rgba(167,139,250,.7);transition:width .6s}
.xt-end{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:12px;font-size:12px;color:var(--d)}
.xt-end code{padding:3px 8px;border-radius:7px;border:1px solid var(--line);background:rgba(3,8,20,.8);color:var(--c);font-size:12px;overflow-wrap:anywhere}
.xt-log{margin-top:10px;padding:8px 10px;max-height:94px;overflow:hidden;border-radius:9px;border:1px solid rgba(127,144,173,.18);background:rgba(2,5,12,.85);font-size:11px;line-height:1.55;color:#8ea2c2}
.xt-log div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xt-log div:last-child::after{content:"▋";margin-left:3px;color:var(--c);animation:xt-blink 1s steps(2) infinite}
.xt-msg{margin-top:10px;font-size:12.5px;color:var(--d)}
.xt-msg.err{color:var(--r)}.xt-msg.ok{color:var(--g)}
.xt-foot{padding:0 22px 14px;font-size:11.5px;color:#5d6b86;letter-spacing:.04em}
/* by the console's own width (a host page may give it less than the window) */
@container xt (max-width:1199px){.xt-body{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}.xt-rail{grid-column:1/-1}
.xt-models{grid-template-columns:repeat(auto-fill,minmax(210px,1fr));max-height:none;margin-right:0;padding-right:0}}
@container xt (max-width:819px){.xt-body{grid-template-columns:minmax(0,1fr)}}
@container xt (max-width:559px){.xt-head{flex-direction:column;align-items:flex-start;padding:16px}.xt-state{max-width:100%}.xt-body{padding-left:12px;padding-right:12px}
.xt-models,.xt-grid{grid-template-columns:minmax(0,1fr)}.xt-hud{grid-template-columns:minmax(0,1fr);justify-items:center}.xt-hud>div+div{width:100%}
.xt-title span{display:block;margin:4px 0 0}.xt-actions .xt-btn{flex:1 1 auto;justify-content:center}.xt-foot{padding:0 14px 14px}}
@media (prefers-reduced-motion:reduce){.xt::after,.xt-engage::after,.xt-logo .spin,.xt-led,.xt-log div:last-child::after{animation:none!important}}
"""

LOGO = (
    "<svg class=xt-logo viewBox='0 0 48 48' aria-hidden=true>"
    "<defs><linearGradient id=xt-lg x1=0 y1=0 x2=1 y2=1><stop offset=0 stop-color='#3ee6ff'/><stop offset=1 stop-color='#a78bfa'/></linearGradient></defs>"
    "<polygon points='24,3 42,13.5 42,34.5 24,45 6,34.5 6,13.5' fill='rgba(62,230,255,.08)' stroke='url(#xt-lg)' stroke-width='2'/>"
    "<g class=spin><circle cx=24 cy=24 r=13 fill=none stroke='#3ee6ff' stroke-width='1.2' stroke-dasharray='3 4' opacity='.8'/></g>"
    "<path d='M16 16l16 16M32 16L16 32' stroke='url(#xt-lg)' stroke-width='3.2' stroke-linecap=round/>"
    "<circle cx=24 cy=24 r=3 fill='#e9fbff'/></svg>"
)

PHASES = [('boot', 'BOOT'), ('weights', '权重'), ('draft', '草稿'), ('pool', '专家池'), ('graphs', 'CUDA 图'), ('online', 'ONLINE')]


def card_html(data):
    """The console card for data = Console.console_data() + 'status' (Console.status()), optionally 'texts'
    (subtitle, foot, confirm, endpoint) and 'api' (start, stop, status, job: paths relative to the page)."""
    texts = data.get('texts') or {}
    payload = html.escape(json.dumps(data, ensure_ascii=False), quote=True)
    phases = ''.join(f'<li data-phase={key}>{label}</li>' for key, label in PHASES)
    return f"""<section class=xt id=xtier-console data-state=offline data-xtier="{payload}" aria-label="XTier 启动控制台">
<header class=xt-head>
  <div class=xt-brand>{LOGO}<div><div class=xt-title>XTIER<span>HYBRID MoE CONSOLE</span></div>
  <div class=xt-sub>{html.escape(texts.get('subtitle', SUBTITLE))}</div></div></div>
  <div class=xt-state><span class=xt-led></span><b id=xt-state>OFFLINE</b><small id=xt-state-sub>待命</small></div>
</header>
<div class=xt-body>
  <div class="xt-panel xt-rail" role=group aria-label="XTier 模型">
    <h4>MODELS · 模型</h4>
    <div class=xt-legend>README 实测 tok/s · 文本：64K / 512K 提示 · AGENT：编码代理会话（悬停看测法）</div>
    <div class=xt-models role=tablist></div>
  </div>
  <form class=xt-panel id=xt-form autocomplete=off>
    <h4>LAUNCH PARAMETERS · 启动参数</h4>
    <div class=xt-sel id=xt-sel></div>
    <div class=xt-grid id=xt-options></div>
    <div class=xt-notes id=xt-notes></div>
    <div class=xt-actions>
      <button type=submit class="xt-btn xt-engage" id=xt-engage>启动 · ENGAGE</button>
      <button type=button class="xt-btn xt-abort" id=xt-abort disabled>停止 · ABORT</button>
      <button type=button class=xt-btn id=xt-reset>恢复实测默认</button>
    </div>
    <div class=xt-msg id=xt-msg aria-live=polite></div>
  </form>
  <div class=xt-panel>
    <h4>TELEMETRY · 运行状态</h4>
    <div class=xt-hud>
      <div class=xt-ring>
        <svg viewBox='0 0 150 150' aria-hidden=true><defs><linearGradient id=xt-grad x1=0 y1=0 x2=1 y2=1><stop offset=0 stop-color='#3ee6ff'/><stop offset=.6 stop-color='#a78bfa'/><stop offset=1 stop-color='#f472b6'/></linearGradient></defs>
        <circle class=track cx=75 cy=75 r=62 /><circle class=ticks cx=75 cy=75 r=71 /><circle class=bar id=xt-arc cx=75 cy=75 r=62 stroke-dasharray='389.6' stroke-dashoffset='389.6' /></svg>
        <div class=core><b class=xt-num id=xt-core>--</b><span id=xt-core-sub>STANDBY</span></div>
      </div>
      <div>
        <ul class=xt-phases id=xt-phases>{phases}</ul>
        <div class=xt-meters>
          <div class=xt-meter><span>DECODE</span><b class=xt-num id=xt-tps>--</b></div>
          <div class=xt-meter><span>接受长度</span><b class=xt-num id=xt-accept>--</b></div>
          <div class=xt-meter><span>运行请求</span><b class=xt-num id=xt-running>--</b></div>
        </div>
        <div class=xt-spark><svg viewBox='0 0 200 46' preserveAspectRatio=none id=xt-spark><path class=ar d=''/><path class=ln d=''/></svg></div>
      </div>
    </div>
    <div class=xt-gpu><span id=xt-gpu-text>GPU --</span><div class=bar><i id=xt-gpu-bar></i></div></div>
    <div class=xt-end id=xt-end></div>
    <div class="xt-log xt-mono" id=xt-log></div>
  </div>
</div>
<div class=xt-foot>{html.escape(texts.get('foot', FOOT))}</div>
</section>"""


JS = r"""
(() => {
  const root = document.getElementById('xtier-console');
  if (!root) return;
  const data = JSON.parse(root.dataset.xtier);
  const T = data.texts || {};
  const API = Object.assign({start: 'api/start', stop: 'api/stop', status: 'api/status', job: 'api/job'}, data.api || {});
  const hooks = window.xtierConsoleHooks || {};
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fill = (tpl, vals, html) => String(tpl).replace(/\{(\w+)\}/g, (_, k) => html ? esc(vals[k]) : String(vals[k] ?? ''));
  const url = p => { const u = new URL(p, location.href); u.username = ''; u.password = ''; return u.href; };
  const post = (p, body) => fetch(url(p), {method: 'POST', headers: {'Accept': 'application/json', 'X-Requested-With': 'fetch'}, body});
  const matches = (cond, v) => Object.entries(cond || {}).every(([k, x]) => v[k] === x);
  const models = Object.keys(data.models);
  const chosen = {};
  for (const m of models) chosen[m] = {...data.defaults[m], ...(data.saved[m] || {})};
  let model = data.saved.model && data.models[data.saved.model] ? data.saved.model : models[0];
  let live = {state: 'offline'};
  let busy = false;

  const tabs = root.querySelector('.xt-models');
  // bench: the README speeds as text (dec, pre), how they were measured (kind: agent / text) and a tooltip (title)
  tabs.innerHTML = models.map(m => {
    const d = data.models[m], b = d.bench || {};
    const kind = {agent: 'AGENT', text: '文本'}[b.kind] || '';
    const perf = b.dec || b.pre ? `<div class="perf xt-mono"><span>DEC</span><b>${esc(b.dec || '--')}</b>` +
      (kind ? `<i class=kind data-kind="${esc(b.kind)}">${kind}</i>` : '') + `<span>PRE</span><b>${esc(b.pre || '--')}</b></div>` : '';
    const tip = [d.arch, b.title].filter(Boolean).join('\n');
    return `<button type=button class=xt-model role=tab data-model="${esc(m)}" aria-selected=false title="${esc(tip)}"><div>
      <div class=top><span class="code xt-mono">${esc(d.code || m)} · :${esc(d.port)}</span><span class="run xt-mono"></span></div>
      <div class=name>${esc(d.name)}</div>${perf}</div></button>`;
  }).join('');
  tabs.addEventListener('click', e => {
    const t = e.target.closest('.xt-model');
    if (!t) return;
    model = t.dataset.model;
    paintTabs(); renderOptions();
  });

  function paintTabs() {
    for (const t of tabs.querySelectorAll('.xt-model')) {
      const here = live.model === t.dataset.model && (live.state === 'online' || live.state === 'loading');
      t.setAttribute('aria-selected', String(t.dataset.model === model));
      t.dataset.running = here ? live.state : '';
      t.querySelector('.run').textContent = here ? (live.state === 'online' ? 'RUNNING' : 'LOADING') : '';
    }
  }

  function renderOptions() {
    const d = data.models[model];
    $('xt-sel').innerHTML = `<b>${esc(d.name)}</b><span>${esc(d.arch || '')}</span>` +
      `<span class=hw>${d.hardware ? esc(d.hardware) + ' · ' : ''}:${esc(d.port)}</span>`;
    const host = $('xt-options');
    host.innerHTML = data.options[model].map(o => {
      const opts = o.choices.map(([v, label]) => `<option value="${esc(v)}"${chosen[model][o.key] === v ? ' selected' : ''}>${esc(label)}</option>`).join('');
      return `<label class=xt-field data-key="${esc(o.key)}" title="${esc(o.hint || '')}">${esc(o.label)}<select name="${esc(o.key)}">${opts}</select></label>`;
    }).join('');
    for (const sel of host.querySelectorAll('select')) sel.addEventListener('change', () => { chosen[model][sel.name] = sel.value; paintOptions(); });
    paintOptions();
  }

  // an option without effect under the others (off_when) is disabled; notes say what a combination does
  function paintOptions() {
    const v = chosen[model], d = data.defaults[model], notes = [];
    let changed = 0;
    for (const o of data.options[model]) {
      const f = $('xt-options').querySelector(`.xt-field[data-key="${o.key}"]`);
      const off = (o.off_when || []).some(c => matches(c, v));
      f.querySelector('select').disabled = off;
      f.classList.toggle('changed', !off && v[o.key] !== d[o.key]);
      if (!off && v[o.key] !== d[o.key]) changed++;
    }
    for (const n of data.models[model].notes || []) if (matches(n.when, v)) notes.push(n.warn ? `<b>${esc(n.text)}</b>` : esc(n.text));
    notes.push(changed ? `<b>${changed} 项不同于实测默认</b>` : '全部为实测默认配置');
    $('xt-notes').innerHTML = notes.join(' · ');
  }

  $('xt-reset').addEventListener('click', () => { chosen[model] = {...data.defaults[model]}; renderOptions(); });

  const msg = (text, kind = '') => { const el = $('xt-msg'); el.className = 'xt-msg ' + kind; el.textContent = text; };

  $('xt-form').addEventListener('submit', async e => {
    e.preventDefault();
    if (busy) return;
    const d = data.models[model];
    const ask = (hooks.confirmPrefix ? hooks.confirmPrefix() : '') +
      fill(T.confirm || '启动 XTier · {name}？\n会先停掉正在运行的 XTier 模型；就绪后在 127.0.0.1:{port} 提供 OpenAI 兼容接口（模型名 {served}）。', d, false);
    if (!confirm(ask)) return;
    busy = true; $('xt-engage').disabled = true;
    msg('正在提交启动任务…');
    try {
      const res = await post(API.start, new URLSearchParams({model, ...chosen[model]}));
      const out = await res.json().catch(() => ({ok: false, error: '接口返回非 JSON'}));
      if (!res.ok || !out.ok) throw new Error(out.error || `HTTP ${res.status}`);
      msg(`已开始：${out.label}。后台启动中，可以关闭页面。`, 'ok');
      watchJob(out.job);
    } catch (err) {
      msg('启动未提交：' + (err.message || err), 'err');
      busy = false; $('xt-engage').disabled = false;
    }
  });

  async function watchJob(job) {
    for (;;) {
      await new Promise(r => setTimeout(r, 2500));
      let st;
      try { st = await (await fetch(url(API.job), {cache: 'no-store'})).json(); } catch (err) { continue; }
      if (st.id !== job) break;
      if (st.state === 'running') continue;
      if (st.state === 'done') msg(st.message || '已就绪', 'ok'); else msg('启动未完成：' + (st.error || '未知错误'), 'err');
      break;
    }
    busy = false; poll();
  }

  $('xt-abort').addEventListener('click', async () => {
    if (!confirm('停止 XTier 服务？正在加载的启动也会中止。')) return;
    $('xt-abort').disabled = true;
    msg('正在停止…');
    try {
      const res = await post(API.stop);
      const out = await res.json().catch(() => ({ok: false, error: '接口返回非 JSON'}));
      msg(out.ok ? (out.message || '已停止') : ('停止失败：' + (out.error || res.status)), out.ok ? '' : 'err');
    } catch (err) { msg('停止失败：' + (err.message || err), 'err'); }
    poll();
  });

  function spark(values) {
    const svg = $('xt-spark');
    if (!values || values.length < 2) { svg.querySelector('.ln').setAttribute('d', ''); svg.querySelector('.ar').setAttribute('d', ''); return; }
    const hi = Math.max(...values), lo = Math.min(...values), span = Math.max(hi - lo, hi * 0.05, 1), n = values.length;
    const pts = values.map((v, i) => [i / (n - 1) * 200, 40 - (v - lo) / span * 32]);
    const line = 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
    svg.querySelector('.ln').setAttribute('d', line);
    svg.querySelector('.ar').setAttribute('d', line + ' L200 46 L0 46 Z');
  }

  function render(s) {
    live = s;
    root.dataset.state = s.state;
    const states = {offline: ['OFFLINE', '待命'], loading: ['BOOTING', s.phase_label || '启动中'], online: ['ONLINE', 'XTier 在线'], failed: ['FAULT', '上次启动失败']};
    const [big, sub] = states[s.state] || states.offline;
    $('xt-state').textContent = big;
    const mins = s.elapsed ? `${Math.floor(s.elapsed / 60)}:${String(s.elapsed % 60).padStart(2, '0')}` : '';
    $('xt-state-sub').textContent = s.state === 'offline' ? sub : `${data.models[s.model]?.name || ''} · ${sub}${mins ? ' · ' + mins : ''}`;
    const pct = s.state === 'online' ? 100 : (s.progress || 0);
    $('xt-arc').setAttribute('stroke-dashoffset', String(389.6 * (1 - pct / 100)));
    $('xt-core').textContent = s.state === 'online' ? (s.tps ? Math.round(s.tps) : 'ON') : (s.state === 'loading' ? pct + '%' : '--');
    $('xt-core-sub').textContent = s.state === 'online' ? (s.tps ? 'TOK / S' : 'READY') : (s.state === 'loading' ? 'LOADING' : (s.state === 'failed' ? 'FAULT' : 'STANDBY'));
    const order = ['boot', 'weights', 'draft', 'pool', 'graphs', 'online'];
    const at = order.indexOf(s.state === 'online' ? 'online' : (s.phase || ''));
    for (const li of $('xt-phases').children) {
      const i = order.indexOf(li.dataset.phase);
      li.className = at < 0 ? '' : (i < at ? 'done' : (i === at ? 'now' : ''));
    }
    $('xt-tps').innerHTML = s.tps ? `${s.tps.toFixed(1)}<small>tok/s</small>` : '--';
    $('xt-accept').textContent = s.accept ? s.accept.toFixed(2) : '--';
    $('xt-running').textContent = s.state === 'online' ? String(s.running ?? 0) : '--';
    spark(s.spark);
    const g = s.gpu || {};
    $('xt-gpu-text').textContent = g.mem_total ? `${g.name || 'GPU'}${g.count > 1 ? ' ×' + g.count : ''} · 显存 ${g.mem_used} / ${g.mem_total} GiB · 利用率 ${g.util}% · ${g.power} W` : 'GPU --';
    $('xt-gpu-bar').style.width = g.mem_total ? (g.mem_used / g.mem_total * 100).toFixed(1) + '%' : '0';
    if (s.state === 'online' || s.state === 'loading') {
      const opts = s.options ? Object.entries(s.options).filter(([k, v]) => data.defaults[s.model] && data.defaults[s.model][k] !== v).length : 0;
      $('xt-end').innerHTML = fill(T.endpoint || '<span>OpenAI 兼容接口</span><code>http://127.0.0.1:{port}/v1</code><span>模型名</span><code>{served}</code>', s, true) +
        (s.version ? `<span>XTier</span><code>${esc(s.version)}</code>` : '') + (s.load_s ? `<span>加载用时 ${esc(s.load_s)} s</span>` : '') +
        `<span>${opts ? opts + ' 项非默认参数' : '实测默认参数'}</span>`;
    } else if (s.state === 'failed') {
      $('xt-end').innerHTML = `<span>${esc(s.error || '')}</span>`;
    } else {
      $('xt-end').innerHTML = '<span>未运行 · 选择模型后点击启动</span>';
    }
    $('xt-log').innerHTML = (s.tail || []).map(l => `<div>${esc(l)}</div>`).join('') || '<div>— 等待指令 —</div>';
    const runningHere = s.state === 'loading' || s.state === 'online';
    $('xt-abort').disabled = !runningHere;
    $('xt-engage').disabled = busy || s.state === 'loading';
    $('xt-engage').textContent = runningHere && s.model === model ? '重启 · 应用参数' : '启动 · ENGAGE';
    paintTabs();
  }

  let timer = null;
  async function poll() {
    clearTimeout(timer);
    try {
      const res = await fetch(url(API.status), {cache: 'no-store'});
      if (res.ok) render(await res.json());
    } catch (err) { /* keep the last picture */ }
    const fast = live.state === 'loading' || live.state === 'online' || busy;
    timer = setTimeout(poll, document.hidden ? 15000 : (fast ? 2000 : 8000));
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  if (data.status && ['online', 'loading'].includes(data.status.state) && data.models[data.status.model]) model = data.status.model;
  paintTabs(); renderOptions();
  if (data.status) render(data.status);
  poll();
})();
"""


def page_html(data, title='XTier Console'):
    """The standalone page of `python -m xtier.console`."""
    return ('<!doctype html><html lang=zh-CN><head><meta charset=utf-8>'
            '<meta name=viewport content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>html,body{{margin:0;background:#03050c;color:#dcecff}}'
            'body{padding:24px 16px}.xt-page{max-width:1360px;margin:0 auto}' + CSS + '</style></head><body>'
            f'<main class=xt-page>{card_html(data)}</main><script>{JS}</script></body></html>')
