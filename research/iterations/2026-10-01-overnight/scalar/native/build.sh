#!/usr/bin/env bash
# Build the wnn_cpp Python extension.
#   native_cpp/build.sh PYTHON [LEVEL ...]      LEVEL in v2 | avx2 | v3 (default: v2 v3)
# Output: native_cpp/build/<LEVEL>/wnn_cpp<EXT_SUFFIX>. libstdc++/libgcc are linked statically so the
# module only needs glibc from the scorer image (python:3.11-slim-bookworm); check_glibc.sh verifies
# that no symbol needs a glibc newer than bookworm's 2.36.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${1:?python interpreter with pybind11 installed}
shift
levels=("${@:-v2 v3}")
read -r -a levels <<< "${levels[*]}"
suffix=$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
includes=$("$py" -m pybind11 --includes)
for level in "${levels[@]}"; do
  case $level in
    v2) march="-march=x86-64-v2" ;;
    avx2) march="-march=x86-64-v2 -mavx2 -mfma" ;;
    v3) march="-march=x86-64-v3" ;;
    *) echo "unknown level $level" >&2; exit 2 ;;
  esac
  out="$here/build/$level"
  mkdir -p "$out"
  gcc -O2 -std=c11 -fPIC -c "$here/src/glibc_compat.c" -o "$out/glibc_compat.o"
  # no-trapping-math / no-math-errno do not change any result; they let GCC if-convert the float
  # clamps in the activations so those loops vectorise
  g++ -O3 $march -mtune=generic -fno-trapping-math -fno-math-errno -std=c++17 -shared -fPIC -fvisibility=hidden \
      -static-libstdc++ -static-libgcc $includes -I"$here/include" \
      "$here/src/bindings.cpp" "$out/glibc_compat.o" -o "$out/wnn_cpp$suffix"
  rm -f "$out/glibc_compat.o"
  max_glibc=$(objdump -T "$out/wnn_cpp$suffix" | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
  echo "$level needs up to $max_glibc (scorer image: GLIBC_2.36)"
  echo "built $out/wnn_cpp$suffix"
done
