# XTier 规划器

决定热专家怎么放：哪些 MoE 层把专家全部放进显存（全常驻），其余每层放几个热专家槽（没命中的由 CPU 引擎算）。
目标是同一份显存预算下 decode 最快：每层的 GPU 部分（命中的专家）和 CPU 部分（没命中的）同时算，谁慢谁决定这一层的时间，
所以要按"两边同时算完"分配；在不让 decode 变慢的前提下，多出来的自由度用来多放全常驻层（prefill 不用再上传这些层）。

## 流程

```
tools/plan_probe.py   本机 × 层形状的耗时系数（约 1–3 分钟，GPU 和 CPU 专家线程要空闲）
tools/plan_trace.py   路由档案（每个专家被多少个解码窗口用到、多少个 token 路由到它）
tools/plan_auto.py    方案：打印与手工布局的对比，写方案文件
XTIER_CACHE_PLAN=文件 | auto   启动时按方案建缓存
tools/plan_calibrate.py        用一次运行的缓存日志校准（自适应系数、CPU 侧）
```

### 1. 探测

```bash
python tools/plan_probe.py --preset qwen38 --threads 112 --out probe.json   # 预设见 tests/layers.py
python tools/plan_probe.py --shape 512,2560,640,10,nvfp4 --m 4 ...          # 或直接给 E,H,I,top-k,格式
```

按层形状造合成权重（随机 FP4 码、scale 为 1，内核速度与数值无关，不需要检查点），走 XTier 自己的解码路径（信箱、CUDA 图），
每张图 32 次调用、每次换一组路由（权重从显存 / 内存读，不是从缓存读，和真实 decode 一样）。测：

| 项 | 路径 | 拟合 |
|---|---|---|
| gpu(n) | 全常驻层，n 个不同专家 | g0 + g1·n（每个用到的专家读一次，与 token 数无关） |
| cpu(n, r, t) | 不缓存的层，n 个专家、r 个路由（token-专家对）、t 个有路由的 token | c0 + c1·n + c2·r + c3·t（专家大时每个路由的计算约是读一次权重的 1/6；c3 在 Qwen 上约 0.8 µs，可忽略） |
| part(g, c) | 部分缓存层，g 个命中、c 个没命中，两边同时算 | max(gpu(g) + p0, cpu(c, r_c) + p1) |
| h2d | prefill 从锁页内存上传专家 | GB/s |

`--m` 是每次解码调用的 token 数（投机解码验证宽度，默认取预设：Qwen / GLM 4、DS 6、MiniMax 8）。

### 2. 路由档案

LKQstrata 档案格式（每层专家按使用次数排序）加 `counts`（每个专家被多少个解码窗口用到）和 `windows`；`routes`（每个专家的
token 数）可选，没有时按窗口内平均每个专家的 token 数估。Qwen 生产档案（`lkqstrata-qwen38.json`）和 DS 种子档案带 counts。
别的模型用路由记录生成：

```bash
XTIER_ROUTE_TRACE=trace.bin <启动脚本> ...      # 跑一段有代表性的负载
python tools/plan_trace.py trace.bin --log 服务日志 --experts 288 --out profile.json
```

记录的是每次 CPU 解码调用的原始路由（命中缓存的也在内），只覆盖经信箱的层（部分缓存或不缓存的层），全常驻层不会出现。

### 3. 方案

```bash
python tools/plan_auto.py --profile profile.json --probe probe.json --layout "0,1,2,5:155" --adapt 1.6 \
    [--keep-decode] [--tol 0.003] [--layers] [--write plan.json]
```

- `--layout`：手工布局（`XTIER_CACHE_FULL` 列表 : `XTIER_CACHE_SLOTS`），默认预算 = 它在档案各层上的槽数（同显存）；
  也可以 `--budget 槽数` / `--budget-gib`。
- 输出手工布局、decode 最快的方案、选定方案的每窗口 MoE 耗时和每个 prefill 块的上传时间；`--layers` 逐层对比。
- 选定方案：decode 不超过最快方案 ×(1+tol)（`--keep-decode`：不超过手工布局的 decode，且上传不多于手工布局）的方案里，
  上传最少（全常驻层最多）的那个。启动时规划默认就是 keep（`XTIER_PLANNER_KEEP=1`）。
- 保护（`--min-gain`，默认 0.01）：方案的上传不比手工布局少、decode 预测收益又不到 1% 时，保留手工布局（预算多出的槽平均
  加到部分缓存层）。GLM 实测：模型预测不均匀分槽快 0.8%，实际没有，逐层模型没这么准，小的挪动不值得。
- `--add-gib`：在手工布局的预算上再加（例如测得的 prefill 峰值之外的空余显存）。
- `--adapt`：自适应换入让 S 个槽命中得像档案前 adapt×S 名。用 `plan_calibrate.py` 从一次运行求（DS 1.69、GLM 1.47、MiniMax 2.19、
  Qwen 约 1.4–1.6）。
- `--call-us`：真实服务里每次信箱调用比单层探测多的微秒数。本机 Qwen 用五组实测布局标定为 28（配 v3 探测、adapt 1.4），来源未定位
  （不在 CPU 引擎里：服务内引擎耗时与探测相同；调用之间的 GPU 工作约 +11 µs、多层切换约 +5 µs；槽位扫描、token 数、统计锁、
  主线程 CPU 忙等都不是）。

### 4. 启动

- `XTIER_CACHE_PLAN=plan.json`：方案里列出的层用方案的槽数（等于专家数 = 全常驻），没列出的层仍按
  `XTIER_CACHE_FULL` / `XTIER_CACHE_SLOTS`。`XTIER_CACHE_PROFILE` 照常决定开机先放哪些专家。
- `XTIER_CACHE_PLAN=auto`：启动时现算（`xtier/planner/auto.py from_env`），需要 `XTIER_CACHE_PROFILE`（带 counts）和
  `XTIER_PLANNER_PROBE`；预算默认 = 启动脚本里的手工布局（同显存，`XTIER_CACHE_BUDGET` / `XTIER_CACHE_BUDGET_GIB` 可改，
  `XTIER_CACHE_BUDGET_ADD_GIB` 在其上加）；默认 `XTIER_PLANNER_KEEP=1`：decode 不差于手工布局、上传不多于手工布局（`0` 按最快方案），`XTIER_PLANNER_TOL`、
  `XTIER_PLANNER_LAM`、`XTIER_PLANNER_ADAPT`、`XTIER_PLANNER_CALL_US`、`XTIER_PLANNER_MIN_GAIN` 同命令行。日志里有一行 `xtier planner:`（全常驻层、
  槽数、预测的 decode 和上传，保留手工布局时注明）。

### 5. 校准

```bash
python tools/plan_calibrate.py --profile profile.json --log 服务日志 --layout ":60" --probe probe.json
```

读运行日志里的 `XTier cache:` 行（`XTIER_CACHE_LOG`，所有部分缓存层合计的命中率、每次调用的专家数和 CPU 耗时），求自适应
系数，并对照模型预测的 CPU 侧耗时。注意这一行只在第一个 XTier 层经信箱时才打印（Qwen 第 0 层全常驻，所以没有）。

## 模型

每个解码窗口（一次验证前向），一层：

```
全常驻          gpu(n_g)
部分缓存        E[max(gpu(n_g) + p0, cpu(n_c, r_c, t_c) + p1)]
不缓存          cpu(n_c, r_c, t_c)
```

n_g / n_c 是命中 / 没命中的不同专家数：档案里使用率为 p 的专家贡献均值 p、方差 p(1−p)，槽里放使用最多的 adapt×S 个；
max 的期望用 Clark 公式（两边接近时窗口间的波动要多付一点）。prefill：不是全常驻的层每个 prefill 块要上传整层专家
（E × 每专家字节 ÷ h2d）。注意力、稠密层、采样等与方案无关，不在模型里。

优化：每层槽数取 0、8、16、…或全部，全局预算上动态规划最小化 decode + λ·上传；λ 从 0 往上扫（网格后二分），decode 随 λ
增大、上传随 λ 减小，取满足 decode 界限的最大 λ。约 1 秒（50 层 × 512 专家）。

## 限制

- 显存预算现在取手工布局（同显存）或给定值；还不会自己算"权重、KV 池、CUDA 图、prefill 分块之后还剩多少"。空余显存的测法：
  跑最长的 prefill（512K）时每 0.5 s 采 `nvidia-smi --query-gpu=memory.used`，峰值之外留 1.5 GiB（GLM、MiniMax 在本机已没有余量）。
- 探测只用一层反复调用，比真实服务每次信箱调用少约 28 µs（`--call-us` 补）；即使补上，离手工布局远的方案仍有约 2% 的误差
  （Qwen 29 层方案 decode 预测 +2.75%，agent 实测 +0.5%）。所以默认只在手工布局附近挪动（KEEP）。
- 只决定槽数；CPU 线程、prefill 分块、并发仍是启动脚本里的值。
- 自适应系数依赖负载和换入节奏（`XTIER_CACHE_EVERY` / `SWAPS`），换了要重新校准。
- 检查：`python tests/planner_check.py`（不需要 GPU）。
