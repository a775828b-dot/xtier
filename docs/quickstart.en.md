# Quick start

From an empty machine to the first request. This route uses Qwen3.8 Flash Next: it needs the least (one 24 GB GPU) and has been verified on the most GPUs ([support table](support.md), in Chinese).

[中文](quickstart.md) · [README](README.en.md)

## What you need

| Item | Requirement |
|---|---|
| GPU | 24 GB or more; run on RTX 4090, RTX 3090 and RTX PRO 5000 |
| Driver | 580 or newer (CUDA 13.0), see `nvidia-smi` |
| Host memory | about 120 GB (about 71 GiB at the peak of serving, more while loading) |
| Disk | 200 GiB or more on an SSD (the model 121 GiB, the environment about 15 GiB) |
| System | Linux, Python 3.12, g++ (tested: Ubuntu 24.04, g++ 13) |

Everything below lives in the directory `DATA` names; set it to your data disk:

```bash
export DATA=/path/to/data
mkdir -p $DATA/src $DATA/models
```

## 1. Environment and PyTorch

```bash
python3.12 -m venv $DATA/xtier-env
P=$DATA/xtier-env/bin/python
$P -m pip install -U pip wheel setuptools cmake ninja setuptools-scm setuptools-rust
$P -m pip install torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0 torchcodec==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu130

# SGLang's kernel builds link libcudart; torch's CUDA wheels only have libcudart.so.13
CU=$DATA/xtier-env/lib/python3.12/site-packages/nvidia/cu13
ln -s libcudart.so.13 $CU/lib/libcudart.so
ln -s lib $CU/lib64
```

## 2. XTier and the tested SGLang

```bash
git clone -b v0.1.3 https://github.com/a775828b-dot/xtier $DATA/src/xtier
git clone https://github.com/sgl-project/sglang $DATA/src/sglang
git -C $DATA/src/sglang checkout 27bc1a0615

SGLANG_BUILD_RUST_EXTS=none $P -m pip install --no-build-isolation \
  -c $DATA/src/xtier/constraints.txt -e $DATA/src/sglang/python
$P -m pip install --no-build-isolation --no-deps -e $DATA/src/xtier

cd $DATA/src/xtier
PATH=$DATA/xtier-env/bin:$PATH tools/build_engine.sh $CU    # CPU engine and mailbox
CUDA_HOME=$CU $P tools/check_sglang.py --strict
```

The last command should print `the same commit` and `0 of ... hook targets and 0 of ... imports missing`.

## 3. The model and its tokenizer

```bash
$DATA/xtier-env/bin/hf download lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4 \
  --revision f748827507ec --local-dir $DATA/models/qwen38      # 121 GiB
```

The checkpoint has no `tokenizer.json`, and current transformers cannot read its tokenizer. Convert it once with an older version, in an environment of its own that you can delete afterwards:

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

## 4. Start the server

```bash
cd $DATA/src/xtier
DATA=$DATA configs/qwen38-rtx4090-3090/run-qwen.sh > $DATA/qwen.log 2>&1 &
tail -f $DATA/qwen.log
```

It is ready at `The server is fired up`. The first start compiles kernels first (the log stays at `building FlashInfer modules`) and then loads the weights: about 6 minutes here (1 of them compiling), then about 3 on later starts, longer with slower disks and CPUs.

**Behind a proxy (`http_proxy` set)**, run `export no_proxy=127.0.0.1,localhost` before starting; otherwise the server's requests to itself and the curl below go through the proxy and return 502 or nothing.

Defaults: 48 hot-expert slots per layer, 128K context, port 30000. For two GPUs add `TP=2 CUDA_VISIBLE_DEVICES=0,1`, for speculative decoding `MTP=1`; the rest is in the [configuration notes](../configs/qwen38-rtx4090-3090/README.md).

## 5. The first request

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
  "max_tokens": 200, "temperature": 0
}'
```

The first request takes about half a minute (some kernels still compile), later ones are quick. If the JSON has an answer in `choices[0].message`, it works (the reasoning is in `reasoning_content`). The API is OpenAI-compatible at `http://127.0.0.1:30000/v1`, so any client can connect.

## 6. Check the results (optional, about 2 minutes)

The model copies source code word for word, and all of it has to match:

```bash
cd $DATA/src/xtier
$P bench/copy_test.py --port 30000 --model Qwen3.8-Flash-Next --ignore-first-indent \
  --source $DATA/src/sglang/python/sglang/srt/managers/scheduler.py --tag first
```

It should print `COPY 0/8 copies differ`. One differing chunk now and then need not be an error: at temperature 0 the top-k kernel's output order depends on a race, which changes floating-point rounding, and a token whose score is nearly tied can flip. To judge accuracy use the mean PPL (`bench/ppl_eval.py`).

To stop the server: `pkill -f "[s]glang.launch_server"`.

## If something goes wrong

The [FAQ](faq.md) is in Chinese, but its headings quote the log messages, so you can search for them. The most common ones:

| Log message | Meaning |
|---|---|
| `is tested with SGLang main` | Another SGLang commit: `tools/check_sglang.py` lists what does not apply. Check out the tested commit |
| `hooks not applied`, `has no run_scheduler_process` | This SGLang does not match XTier, so the server stops instead of silently losing the speedups. Same fix |
| `settings XTier does not know are ignored` | A misspelled or removed `XTIER_*` setting, see [xtier/config.py](../xtier/config.py) |
| `building FlashInfer modules` | Normal on a first start. If it happens on every start, the `ninja` in `PATH` keeps changing (a venv's and the system's): keep `PATH` the same |
| `post-warmup freeze_gc failed`, a 502 or an empty answer | `http_proxy` is set: `export no_proxy=127.0.0.1,localhost` |
| `mem_guard: MemAvailable ... killing process group` | Host memory ran low; `largest` names the biggest processes. See [cold experts](README.en.md#cold-experts-on-nvme) if the model does not fit |
| `the checkpoint no longer holds its experts` | The checkpoint was punched by `tools/cold_punch.py` and can only load from the cold store: needs `XTIER_COLD=1`, `XTIER_FAST_LOAD=1` and the same `XTIER_COLD_PATHS`. `--restore` writes the experts back |

## Next

- Other models and GPUs: [support table](support.md).
- A web page to pick models and options: [console](README.en.md#console). Most scripts under `configs/` have the test host's paths: change them first.
- Models larger than host memory: [cold experts](README.en.md#cold-experts-on-nvme).
