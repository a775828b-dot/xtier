# 快速开始

从空机器到第一次请求。这条路线用 Qwen3.8 Flash Next：要求最低（一张 24 GB 的卡），也是在不同显卡上验证最多的模型（见[支持情况](support.md)）。

[English](quickstart.en.md) · [文档目录](README.md)

## 需要什么

| 项目 | 要求 |
|---|---|
| 显卡 | 24 GB 以上；RTX 4090、RTX 3090、RTX PRO 5000 跑通过 |
| 驱动 | 580 以上（CUDA 13.0），`nvidia-smi` 看 |
| 内存 | 约 120 GB（运行时峰值约 71 GiB，加载时更高） |
| 硬盘 | SSD 上 200 GiB 以上（模型 121 GiB，环境约 15 GiB） |
| 系统 | Linux、Python 3.12、g++（测试用 Ubuntu 24.04、g++ 13） |

下面所有命令都用 `DATA` 指定的目录，改成你自己的数据盘：

```bash
export DATA=/path/to/data
mkdir -p $DATA/src $DATA/models
```

## 1. 建环境，装 PyTorch

```bash
python3.12 -m venv $DATA/xtier-env
P=$DATA/xtier-env/bin/python
$P -m pip install -U pip wheel setuptools cmake ninja setuptools-scm setuptools-rust
$P -m pip install torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0 torchcodec==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu130

# SGLang 的内核编译要链接 libcudart，torch 带的 CUDA 包里只有 libcudart.so.13
CU=$DATA/xtier-env/lib/python3.12/site-packages/nvidia/cu13
ln -s libcudart.so.13 $CU/lib/libcudart.so
ln -s lib $CU/lib64
```

## 2. 装 XTier 和测试过的 SGLang

```bash
git clone -b v0.1.1 https://github.com/a775828b-dot/xtier $DATA/src/xtier
git clone https://github.com/sgl-project/sglang $DATA/src/sglang
git -C $DATA/src/sglang checkout 6162b07199

SGLANG_BUILD_RUST_EXTS=none $P -m pip install --no-build-isolation \
  -c $DATA/src/xtier/constraints.txt -e $DATA/src/sglang/python
$P -m pip install --no-build-isolation --no-deps -e $DATA/src/xtier

cd $DATA/src/xtier
PATH=$DATA/xtier-env/bin:$PATH tools/build_engine.sh $CU    # CPU 引擎和信箱库
CUDA_HOME=$CU $P tools/check_sglang.py --strict
```

最后一行应该显示 `the same commit` 和 `0 of ... hook targets and 0 of ... imports missing`。

## 3. 下载模型，转换分词器

```bash
$DATA/xtier-env/bin/hf download lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4 \
  --revision f748827507ec --local-dir $DATA/models/qwen38      # 121 GiB
```

这个模型目录没有 `tokenizer.json`，新版 transformers 读不了它的分词器，要用旧版转一份（单独的环境，用完可以删）：

```bash
python3.12 -m venv $DATA/tok-env
$DATA/tok-env/bin/python -m pip install transformers==5.12.0 tiktoken sentencepiece protobuf jinja2
T=$DATA/models/qwen38-tokenizer; mkdir -p $T
$DATA/tok-env/bin/python -c "
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True).save_pretrained(sys.argv[2])" $DATA/models/qwen38 $T
for f in config.json generation_config.json preprocessor_config.json video_preprocessor_config.json; do
  ln -sfn $DATA/models/qwen38/$f $T/$f
done
rm -rf $DATA/tok-env
```

## 4. 启动

```bash
cd $DATA/src/xtier
DATA=$DATA configs/qwen38-rtx4090-3090/run-qwen.sh > $DATA/qwen.log 2>&1 &
tail -f $DATA/qwen.log
```

看到 `The server is fired up` 就可以用了。第一次要先编译内核（日志停在 `building FlashInfer modules`），再加载权重，实测约 6 分钟（其中编译 1 分钟），之后约 3 分钟，硬盘和 CPU 慢的机器会更久。

**用代理上网的话（设了 `http_proxy`）**，启动前先 `export no_proxy=127.0.0.1,localhost`，否则服务发给自己的请求和下面的 curl 都会走代理，返回 502 或空内容。

默认每层 48 个热专家槽、128K 上下文、端口 30000。两张卡加 `TP=2 CUDA_VISIBLE_DEVICES=0,1`，开投机解码加 `MTP=1`，其余见[配置说明](../configs/qwen38-rtx4090-3090/README.md)。

## 5. 第一次请求

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
  "max_tokens": 200, "temperature": 0
}'
```

第一次请求要等半分钟左右（还有内核要现编），之后就快了。返回 JSON，`choices[0].message` 里有回答就通了（思考过程在 `reasoning_content`）。服务是 OpenAI 兼容接口，地址 `http://127.0.0.1:30000/v1`，客户端都能接。

## 6. 确认结果是对的（可选，约 2 分钟）

让模型逐字照抄一段代码，全对才算过：

```bash
cd $DATA/src/xtier
$P bench/copy_test.py --port 30000 --model Qwen3.8-Flash-Next --ignore-first-indent \
  --source $DATA/src/sglang/python/sglang/srt/managers/scheduler.py --tag first
```

应该显示 `COPY 0/8 copies differ`。偶尔差 1 段不一定是错，[原因见这里](faq.md#温度-0两次输出不完全一样)。

停止服务：`pkill -f "[s]glang.launch_server"`。

## 接下来

- 起不来或报错：[常见问题](faq.md)。
- 用网页选模型和参数：[启动台](console.md)。注意 `configs/` 大多数脚本里的路径是测试机的，要先改。
- 其他模型和显卡：[支持情况](support.md)。
- 内存放不下的大模型：[冷专家](../README.md#冷专家卸载到硬盘)。
