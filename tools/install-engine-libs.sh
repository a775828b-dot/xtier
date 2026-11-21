#!/bin/bash
# 原子安装 XTier 的引擎 .so：先写临时文件，再 rename（同一文件系统上的 rename 是原子的）。
# 直接 `cp` 覆盖会在 O_TRUNC 与写完之间留下一个残缺的文件；此时若有进程正好 dlopen 它
# （服务运行中换内核就是这种情况），会读到半个 .so 而 SIGSEGV。
#
# 用法: install-engine-libs.sh [build-dir]
set -e
BUILD=${1:-/opt/Xtier/build/engine}
DEST=/opt/Xtier/xtier/engine/lib

for so in libxtier_engine.so libxtier_engine_avx512.so libxtier_engine_avx2.so \
          libxtier_mailbox.so libxtier_tables.so; do
  [ -f "$BUILD/$so" ] || { echo "skip $so (not built)"; continue; }
  tmp="$DEST/.${so}.new.$$"
  cp "$BUILD/$so" "$tmp"
  chmod --reference="$DEST/$so" "$tmp" 2>/dev/null || chmod 755 "$tmp"
  mv -f "$tmp" "$DEST/$so"        # 原子替换
  echo "installed $so"
done
