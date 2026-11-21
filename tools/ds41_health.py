#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash (XTier on L20) 一键健康巡检。

用法（在服务器上）：
    /opt/miniforge3/envs/SGLang/bin/python /opt/Xtier/tools/ds41_health.py

输出：服务状态 / 生效配置 / GPU 显存 / NUMA 内存 / 并发 / 近期错误 / elastic 行为 / 缓存命中
只读，不修改任何东西。
"""
import os, re, subprocess, sys, time

LOG = "/var/log/llm/DeepSeek-V4.1-Flash.log"
LIVE = "/opt/Xtier/run-ds41-l20-tp2.sh"
UNIT = "/etc/systemd/system/DeepSeek-V4.1-Flash.service"


def sh(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    section("1. 服务状态")
    print("时间          :", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("systemd       :", sh("systemctl is-active DeepSeek-V4.1-Flash.service"))
    print("NRestarts     :", sh("systemctl show DeepSeek-V4.1-Flash.service -p NRestarts --value"))
    print("进程          :", (sh("pgrep -af 'sglang.launch_server' | head -1") or "<无>")[:120])
    print("Uptime        :", sh("ps -eo etime,comm | grep -m2 'sglang::schedul' | tr '\\n' ' '"))

    section("2. 生效的关键配置（live 脚本 + unit Environment）")
    keys = [
        "XTIER_CACHE_SLOTS", "XTIER_CACHE_ELASTIC", "XTIER_CACHE_ELASTIC_MAX_SLOTS",
        "XTIER_CACHE_ELASTIC_BASE", "XTIER_CACHE_ELASTIC_RESERVE_GIB",
        "XTIER_CACHE_ELASTIC_CRITICAL_GIB", "XTIER_THREADS", "XTIER_TP_NODES",
        "NCCL_P2P_LEVEL", "CUDA_VISIBLE_DEVICES",
    ]
    src = ""
    for f in (LIVE,):
        try:
            src += open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            pass
    for k in keys:
        m = re.findall(rf"{k}=([^ \n]+)", src)
        # 取脚本里的默认值形式 ${K:-value}
        m2 = re.findall(rf"{k}=\$\{{{k}:-([^}}]+)\}}", src)
        val = m2[0] if m2 else (m[0] if m else "<未设置>")
        print(f"  {k:38s} = {val}")
    print()
    print("  命令行参数：")
    for ln in sh(f"grep -E '^  --' {LIVE}").splitlines():
        print("   ", ln.strip())
    print()
    print("  unit Environment：")
    for ln in sh(f"grep 'Environment=' {UNIT}").splitlines():
        print("   ", ln.strip())

    section("3. GPU / NUMA 资源")
    print(sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
             "--format=csv,noheader"))
    print()
    print("GPU 拓扑：")
    print(sh("nvidia-smi topo -m 2>/dev/null | head -9"))
    print()
    print("NUMA 内存：")
    print(sh("numastat -m 2>/dev/null | sed -n '5,9p'"))
    print()
    print("大内存进程（RSS 前 6）：")
    print(sh("ps -eo pid,psr,rss,pcpu,comm --sort=-rss | head -7"))

    section("4. 当前并发 / 缓存")
    m = sh("curl -s --max-time 5 http://127.0.0.1:8001/metrics")
    for ln in m.splitlines():
        if any(k in ln for k in ("num_running_reqs", "num_queue_reqs", "cache_hit_rate", "token_usage")):
            if not ln.startswith("#"):
                print("  ", ln[:150])

    section("5. 最近 30 分钟的错误（应为空）")
    err = sh(f"journalctl -u DeepSeek-V4.1-Flash --since '-30 min' --no-pager 2>/dev/null "
             f"| grep -aiE 'Scheduler hit|SIGQUIT|out of memory|ran twice|Traceback|Error' "
             f"| tail -8 | cut -c1-180")
    print(err or "  (无)")

    section("6. elastic cache 行为（最近 8 条）")
    el = sh(f"grep -a 'elastic' {LOG} 2>/dev/null | tail -8 | cut -c1-190")
    print(el or "  (无)")

    section("7. 启动后显存余量（最后 4 条）")
    print(sh(f"grep -a 'avail mem' {LOG} 2>/dev/null | tail -4 | cut -c1-190"))

    section("8. 最近 Decode 批（accept len / 吞吐）")
    print(sh(f"grep -a 'Decode batch' {LOG} 2>/dev/null | tail -6 | cut -c1-230"))

    section("9. 最近 Prefill 批（看 chunk 规模与 cached 命中）")
    print(sh(f"grep -a 'Prefill batch' {LOG} 2>/dev/null | tail -6 | cut -c1-230"))

    section("9b. 邻居服务（会抢内存带宽的进程）")
    print("  GPU 占用：")
    print(sh("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader"))
    print("  相关 unit：")
    for u in ("Qwen3.8-Flash-Next-XTier.service", "Qwen3.8-Flash-Next.service"):
        st = sh(f"systemctl is-active {u} 2>/dev/null")
        if st and st != "inactive" and st != "unknown":
            print(f"    {u}: {st}")
    print("  node1 上最大的 3 个进程（与 rank1 抢带宽）：")
    print(sh("ps -eo pid,psr,rss,pcpu,comm --sort=-rss | head -4 | tail -3"))

    section("10. 判定")
    health = sh("curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/health")
    nrestart = sh("systemctl show DeepSeek-V4.1-Flash.service -p NRestarts --value")
    mem = sh("grep -a 'avail mem' /var/log/llm/DeepSeek-V4.1-Flash.log 2>/dev/null | tail -1 "
             "| grep -oE 'avail mem=[0-9.]+ GB'")
    ok = True
    if health != "200":
        print("  ✗ /health 不通（服务可能正在重启）"); ok = False
    else:
        print("  ✓ /health = 200")
    if nrestart not in ("0", ""):
        print(f"  ✗ NRestarts={nrestart}（服务重启过，查第 5 节）"); ok = False
    else:
        print("  ✓ NRestarts=0")
    print(f"  · {mem or 'avail mem 未知'}")
    print()
    print("  结论：", "✅ 正常" if ok else "⚠️ 需要处理")


if __name__ == "__main__":
    main()
