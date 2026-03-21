#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${ROOT_DIR}/runtime/instability"
PID_FILE="${RUNTIME_DIR}/host-watch.pid"
LATEST_LINK="${RUNTIME_DIR}/latest.log"
STDOUT_LOG="${RUNTIME_DIR}/host-watch.stdout.log"
INTERVAL="${HOST_WATCH_INTERVAL:-30}"
GPU_TOP="${HOST_WATCH_GPU_TOP:-0}"
UNIT_NAME="host-watch.service"

mkdir -p "${RUNTIME_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

log_header() {
  local log_file="$1"
  {
    echo "==== host-watch start $(timestamp) ===="
    echo "host=$(hostname)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
    echo "kernel=$(uname -a)"
    echo "interval_s=${INTERVAL}"
    echo "gpu_top=${GPU_TOP}"
    echo
  } >> "${log_file}"
}

sample_once() {
  local log_file="$1"
  {
    echo "==== sample $(timestamp) ===="
    echo "-- uptime --"
    uptime
    echo
    echo "-- loadavg --"
    cat /proc/loadavg
    echo
    echo "-- memory --"
    free -h
    echo
    echo "-- psi --"
    for psi in /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io; do
      echo "[$psi]"
      cat "$psi"
    done
    echo
    echo "-- top cpu --"
    ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pcpu | sed -n '1,25p' || true
    echo
    echo "-- top mem --"
    ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pmem | sed -n '1,25p' || true
    echo
    if command -v sensors >/dev/null 2>&1; then
      echo "-- sensors --"
      sensors || true
      echo
    fi
    if [[ "${GPU_TOP}" == "1" ]] && command -v intel_gpu_top >/dev/null 2>&1; then
      echo "-- intel_gpu_top --"
      timeout 3s intel_gpu_top -J -s 100 -o - 2>/dev/null || true
      echo
    fi
    echo "-- recent kernel --"
    journalctl -k -n 40 --no-pager || true
    echo
  } >> "${log_file}"
}

run_loop() {
  local log_file="$1"
  trap 'echo "==== host-watch stop $(timestamp) ====" >> "'"${log_file}"'"; rm -f "'"${PID_FILE}"'"' EXIT INT TERM
  echo "$$" > "${PID_FILE}"
  while true; do
    sample_once "${log_file}"
    sleep "${INTERVAL}"
  done
}

start_watch() {
  local log_file="${RUNTIME_DIR}/host-watch-$(date '+%Y%m%d-%H%M%S').log"
  : > "${STDOUT_LOG}"
  log_header "${log_file}"
  ln -sfn "${log_file}" "${LATEST_LINK}"
  if command -v systemd-run >/dev/null 2>&1 && systemctl --user is-active default.target >/dev/null 2>&1; then
    systemctl --user stop "${UNIT_NAME}" >/dev/null 2>&1 || true
    systemd-run --user --unit="${UNIT_NAME%.service}" \
      --property=WorkingDirectory="${ROOT_DIR}" \
      --setenv=HOST_WATCH_INTERVAL="${INTERVAL}" \
      --setenv=HOST_WATCH_GPU_TOP="${GPU_TOP}" \
      "${BASH_SOURCE[0]}" _run "${log_file}" >/dev/null
    sleep 1
    if systemctl --user is-active "${UNIT_NAME}" >/dev/null 2>&1; then
      echo "started unit=${UNIT_NAME} log=${log_file}"
      exit 0
    fi
  fi

  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "already running pid=$(cat "${PID_FILE}")"
    exit 0
  fi

  bash "${BASH_SOURCE[0]}" _run "${log_file}" </dev/null >> "${STDOUT_LOG}" 2>&1 &
  disown || true
  sleep 1
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "started pid=$(cat "${PID_FILE}") log=${log_file}"
  else
    echo "failed to start"
    exit 1
  fi
}

stop_watch() {
  if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active "${UNIT_NAME}" >/dev/null 2>&1; then
    systemctl --user stop "${UNIT_NAME}" >/dev/null 2>&1 || true
    echo "stopped unit=${UNIT_NAME}"
    rm -f "${PID_FILE}"
    exit 0
  fi
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "not running"
    exit 0
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "stopped pid=${pid}"
  else
    echo "stale pid file pid=${pid}"
  fi
  rm -f "${PID_FILE}"
}

status_watch() {
  if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active "${UNIT_NAME}" >/dev/null 2>&1; then
    echo "running unit=${UNIT_NAME}"
    if [[ -L "${LATEST_LINK}" ]]; then
      echo "log=$(readlink -f "${LATEST_LINK}")"
    fi
    exit 0
  fi
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "running pid=$(cat "${PID_FILE}")"
    if [[ -L "${LATEST_LINK}" ]]; then
      echo "log=$(readlink -f "${LATEST_LINK}")"
    fi
  else
    echo "not running"
    if [[ -L "${LATEST_LINK}" ]]; then
      echo "last_log=$(readlink -f "${LATEST_LINK}")"
    fi
  fi
}

case "${1:-}" in
  start)
    start_watch
    ;;
  stop)
    stop_watch
    ;;
  status)
    status_watch
    ;;
  sample)
    log_file="${2:-${RUNTIME_DIR}/host-watch-manual-$(date '+%Y%m%d-%H%M%S').log}"
    log_header "${log_file}"
    sample_once "${log_file}"
    echo "wrote ${log_file}"
    ;;
  _run)
    run_loop "${2:?log file required}"
    ;;
  *)
    echo "usage: $0 {start|stop|status|sample}"
    exit 2
    ;;
esac
