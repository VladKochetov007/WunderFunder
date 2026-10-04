#!/usr/bin/env bash
set -euo pipefail
control=$1
phase=$2
shift 2
group=$(cut -d: -f3 /proc/self/cgroup)
printf '%s\n' "$group" > "$control/${phase}_scope.txt"
for name in cpu.max memory.max memory.swap.max memory.events; do
  cat "/sys/fs/cgroup$group/$name" > "$control/${phase}_${name}_before.txt"
done
finish() {
  result=$?
  trap - EXIT
  for name in memory.events memory.peak; do
    cat "/sys/fs/cgroup$group/$name" > "$control/${phase}_${name}_after.txt"
  done
  exit "$result"
}
trap finish EXIT
"$@"
