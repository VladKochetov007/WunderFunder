#!/usr/bin/env bash
# Pause only this wave's GPU units and native containers together on sustained heat.
set -u
read_temp() {
  sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4); exit}'
}
live_units() {
  systemctl --user list-units 'wnn-q14-fit-*' 'wnn-q14-export-*' --state=running --no-legend --plain | awk '{print $1}'
}
container_state() {
  podman inspect --format '{{.State.Status}}' "$1" 2>/dev/null || true
}
frozen_units=()
paused_containers=()
for unit in $(live_units); do
  if [[ $(systemctl --user show "$unit" -p FreezerState --value) == frozen ]]; then
    frozen_units+=("$unit")
  fi
done
# Explicit transfer from root's manual pause; do not adopt arbitrary paused containers.
for container in "$@"; do
  if [[ $(container_state "$container") == paused ]]; then paused_containers+=("$container"); fi
done
hot=0
while :; do
  mapfile -t units < <(live_units)
  running_containers=()
  for container in wnn-q14-original-fbc wnn-q14-replacement; do
    if [[ $(container_state "$container") == running ]]; then running_containers+=("$container"); fi
  done
  if (( ${#units[@]} + ${#running_containers[@]} + ${#frozen_units[@]} + ${#paused_containers[@]} == 0 )); then exit 0; fi
  temp=$(read_temp)
  if (( ${#frozen_units[@]} + ${#paused_containers[@]} > 0 )); then
    if [[ -n "$temp" ]] && (( temp <= 84 )); then
      for container in "${paused_containers[@]}"; do podman unpause "$container" || true; done
      for unit in "${frozen_units[@]}"; do systemctl --user thaw "$unit" || true; done
      echo "thaw CPU=$temp units=${frozen_units[*]} containers=${paused_containers[*]}"
      frozen_units=(); paused_containers=(); hot=0
    fi
  else
    if [[ -z "$temp" ]] || (( temp >= 92 )); then hot=$((hot+1)); else hot=0; fi
    if (( hot >= 3 )); then
      for container in "${running_containers[@]}"; do
        if podman pause "$container"; then paused_containers+=("$container"); fi
      done
      for unit in "${units[@]}"; do
        if systemctl --user freeze "$unit"; then frozen_units+=("$unit"); fi
      done
      echo "freeze CPU=${temp:-unavailable} units=${frozen_units[*]} containers=${paused_containers[*]}"
    fi
  fi
  sleep 15
done
