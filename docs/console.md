# XTier 控制台

一个网页管所有模型：选模型、调启动参数（默认是实测配置，改过的项会标出）、启动 / 停止，实时看加载阶段（权重 → 草稿 →
专家池 → CUDA 图 → 在线）和 decode 遥测（tok/s、接受长度、显存）。只用 Python 标准库，不需要 SGLang 或 torch。

![XTier 控制台](images/console.png)

## 运行

```bash
cd xtier                       # 仓库根目录
python3 -m xtier.console       # http://127.0.0.1:8100/
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址。从别的机器访问用 SSH 隧道：`ssh -L 8100:127.0.0.1:8100 主机` |
| `--port` | `8100` | |
| `--config NAME` | 全部 | 只列出 `configs/NAME`（可重复） |
| `--state-dir` | `~/.local/state/xtier-console` | 状态、各模型上次用的参数、模型服务的日志 |
| `--runner` | `auto` | `systemd`：模型服务跑在用户级的临时 scope 里；`setsid`：独立会话，按进程组停止。`auto`：有用户级 systemd 就用它 |
| `--unit` | `xtier-console` | systemd scope 的名字 |
| `--mem-guard-gib` | `20` | 主机可用内存低于这么多 GiB 就停掉模型服务（`tools/mem_guard.sh`；0 = 不守护） |

- 一次只跑一个模型：启动新模型前先停掉当前的，等显存释放；GPU 上别的进程占着超过 2 GiB 就报错、不启动。
- 模型服务在自己的端口上提供 SGLang 的 OpenAI 兼容接口（`http://127.0.0.1:端口/v1`），控制台不转发请求。控制台退出或重启
  不影响正在跑的模型服务，重新打开后接着显示它的状态。
- 页面上的速度数字是 README 测速表里的实测值（64K / 512K 上下文）。

## 加一个模型

在 `configs/<模型>-<硬件>/` 里放启动脚本和 `console.py`，控制台启动时会自动列出它：

```python
from pathlib import Path

from xtier.console.core import common_options

HERE = Path(__file__).resolve().parent

MODEL = {
    'id': 'mymodel', 'order': 40,                 # 唯一的 id；order 决定在列表里的位置
    'name': 'My Model', 'code': 'MY-MODEL', 'port': 18200, 'served': 'My-Model',
    'spec': 'EAGLE', 'arch': '一行结构说明', 'hardware': '一行硬件说明', 'ready_timeout': 1800,
    'bench': {'p64': 0, 'd64': 0.0, 'p512': 0, 'd512': 0.0},   # README 实测 tok/s：prefill / decode × 64K / 512K
    # 公共的四项（加速包、投机解码、上下文、CPU 线程）+ 模型自己的；第一个取值写实测配置
    'options': common_options('EAGLE') + [
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '60',
         'choices': [['60', '60 · 实测'], ['48', '48']],
         'off_when': [{'packs': 'off'}]},       # 这些组合下这一项不起作用，页面上变灰
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：……'}],   # 选中这些组合时显示的说明
}


def build(o, port):
    """o：校验过的参数（都是 choices 里的字符串）；返回 (追加的环境变量, 启动命令)。"""
    env = {'PORT': str(port), 'XTIER_THREADS': o['threads'], 'XTIER_CACHE_SLOTS': o['slots']}
    return env, [str(HERE / 'run-mymodel.sh'), '--context-length', o['context']]


def tags(o):
    """可选：服务标签里附加的说明。"""
    return []
```

- 参数是白名单：页面传来的值不在 `choices` 里就拒绝，`build()` 只拿到校验过的值。
- 默认值写实测过的配置，`bench` 写 README 测速表里的数；实测配置变了，两处一起改。
- `python3 tests/console_check.py` 把每个模型每种参数组合都构建一遍，并检查页面和接口。

## 嵌到别的页面

`xtier.console.core.Console` 是全部逻辑（参数、启动 / 停止、状态），`xtier.console.ui` 的 `card_html(data)`、`CSS`、`JS` 是卡片。
宿主页面可以：

- 用 `data['texts']` 换掉副标题、页脚、启动确认和接口那一行（`subtitle`、`foot`、`confirm`、`endpoint`，可用 `{name}`、`{served}`、
  `{port}` 占位）；
- 用 `data['api']` 指向自己的接口（`start`、`stop`、`status`、`job`，请求和返回的格式同 `xtier/console/__main__.py`）；
- 定义 `window.xtierConsoleHooks.confirmPrefix()`，在启动确认前面加一句话。

## 安全

- 默认只监听 127.0.0.1。监听别的地址时，能访问到它的人都能启动、停止模型服务。
- POST 请求必须带 `X-Requested-With: fetch` 头（跨站表单发不出这个头），别的网页没法借浏览器启动或停止服务。
- 启动模型服务时，从控制台的环境变量里去掉名字含 KEY、TOKEN、SECRET、PASSW 的变量。
