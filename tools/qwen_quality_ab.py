"""质量对照：同一批 prompt 在两种专家格式下（nvfp4 4bit / bf16 无损）的输出比对。

用法：python3 qwen_quality_ab.py out.jsonl     （每种格式各跑一遍，存下输出供比对）
"""
import json
import sys
import time
import urllib.request

URL = 'http://127.0.0.1:8002/v1/chat/completions'
PROMPTS = [
    "用一句话解释 NUMA 亲和性。",
    "写一个 Python 函数，把列表按出现次数排序。",
    "中国的四大发明是什么？各用两句话说明。",
    "计算 17 * 23 + 45 / 5，只给出结果。",
    "用三句话解释什么是张量并行。",
    "把这句话翻译成英文：知识就是力量。",
    "写一句关于秋天的诗。",
    "解释一下 GPU 上 '热专家槽' 是什么意思。",
    "什么是 MoE（混合专家）模型？举一个例子。",
    "列出 Linux 下查看磁盘占用的三条命令。",
    "用一句话说明什么是 FP8 量化。",
    "一道小学数学题：小明有 12 个苹果，给了小红 5 个，还剩几个？",
    "介绍一下 DeepSeek 模型的特点。",
    "写一个 bash 命令，列出当前目录下大于 100MB 的文件。",
    "解释：为什么显存里的数据搬得比内存慢？",
    "用三点列出部署大模型时的显存瓶颈。",
    "介绍一下 AMX 指令集。",
    "写一句 Python 代码：读取 CSV 的第一行。",
    "什么是 KV 缓存？为什么它占显存？",
    "总结一下今天的要点（三条）。",
]


def ask(prompt):
    body = json.dumps({
        'model': 'Qwen3.8-Flash-Next',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 128, 'temperature': 0.0,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out['choices'][0]['message']['content'], time.time() - t0, out['usage']


def main():
    out_path = sys.argv[1]
    rows = []
    for i, p in enumerate(PROMPTS):
        try:
            text, dt, usage = ask(p)
        except Exception as exc:
            text, dt, usage = f'ERROR: {exc}', 0.0, {}
        rows.append({'prompt': p, 'output': text, 'seconds': round(dt, 2),
                     'completion_tokens': usage.get('completion_tokens', 0)})
        print(f'[{i+1:>2}/{len(PROMPTS)}] {dt:5.1f}s  {text[:60].replace(chr(10), " ")}', flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'写入 {out_path}')


if __name__ == '__main__':
    main()
