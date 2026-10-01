#!/usr/bin/env bash
set -euo pipefail
ctx_dir=$(cd "$(dirname "$0")" && pwd)
ctx_repo=$(cd "$ctx_dir/../../../.." && pwd)
ctx_scorer=/home/grinchenkoeg/scorer_image
ctx_pybind=$($ctx_repo/.venv/bin/python -c 'import pybind11;print(pybind11.get_include())')
ctx_suffix=$($ctx_scorer/run.sh -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')
ctx_output=$ctx_dir/native_ctx/build/v3
mkdir -p "$ctx_output"
gcc -O2 -std=c11 -fPIC -c "$ctx_dir/native_ctx/src/glibc_compat.c" -o "$ctx_output/glibc_compat.o"
g++ -O3 -march=x86-64-v3 -mtune=generic -fno-trapping-math -fno-math-errno -std=c++17 -shared -fPIC -fvisibility=hidden \
    -static-libstdc++ -static-libgcc -I"$ctx_scorer/rootfs/usr/local/include/python3.11" -I"$ctx_pybind" \
    -I"$ctx_dir/native_ctx/include" "$ctx_dir/native_ctx/src/bindings.cpp" "$ctx_output/glibc_compat.o" \
    -o "$ctx_output/wnn_cpp$ctx_suffix"
rm "$ctx_output/glibc_compat.o"
echo "Built scorer CPython3.11 native CTX v3 with pybind headers from $ctx_pybind"
