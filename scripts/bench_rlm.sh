#!/usr/bin/env bash
# Compare RLM mode vs baseline on the goose codebase itself.
#
# Both modes use the same model and the same questions. Baseline gives the
# agent shell/file access (default `developer` extension) so it can grep
# through the codebase from cwd. RLM mode pre-loads `crates/goose/src/agents`
# into the per-session store and the model uses rlm__* tools instead.
#
# Usage:
#   GOOSE_BIN=target/release/goose scripts/bench_rlm.sh
#   scripts/bench_rlm.sh "What is LARGE_TEXT_THRESHOLD?"   # single ad-hoc question
#
# Env vars:
#   GOOSE_BIN       — path to goose binary (default: target/release/goose)
#   CONTEXT_PATH    — what to pre-load for --rlm (default: crates/goose/src/agents)
#   CONTEXT_NAME    — name to give it (default: agents)
#   GOOSE_MAX_TURNS — max turns per run (default: 12)

set -euo pipefail

GOOSE_BIN=${GOOSE_BIN:-target/release/goose}
CONTEXT_PATH=${CONTEXT_PATH:-crates/goose/src/agents}
CONTEXT_NAME=${CONTEXT_NAME:-agents}
MAX_TURNS=${GOOSE_MAX_TURNS:-12}

if [[ ! -x "$GOOSE_BIN" ]]; then
  echo "goose binary not found at $GOOSE_BIN — build with: cargo build --release --bin goose" >&2
  exit 1
fi
if [[ ! -e "$CONTEXT_PATH" ]]; then
  echo "context path not found: $CONTEXT_PATH" >&2
  exit 1
fi

# Each question is paired with a (regex) ground-truth pattern. We grep the
# answer; if it matches, the run is "correct". Lenient on purpose — models
# tend to say things like "approximately 13 entries" and that should pass.
declare -a QUESTIONS=(
  "What is the numeric value of the constant LARGE_TEXT_THRESHOLD in the goose codebase? Reply with just the number."
  "What is the value of DEFAULT_MAX_DEPTH for the RLM module? Reply with just the number."
  "Which struct in crates/goose/src/agents/platform_extensions/mod.rs has a field named rlm_store? Reply with just the struct name."
  "How many entries (map.insert calls) are registered in PLATFORM_EXTENSIONS in crates/goose/src/agents/platform_extensions/mod.rs? Reply with just the number."
)
declare -a EXPECTED=(
  "200_?000"
  "\\b2\\b"
  "PlatformExtensionContext"
  "\\b13\\b"
)

# Single-question mode: bench just one
if [[ $# -ge 1 ]]; then
  QUESTIONS=("$1")
  EXPECTED=("${2:-.*}")
fi

BENCH_DIR=$(mktemp -d -t goose-rlm-bench-XXXX)
echo "bench dir: $BENCH_DIR"
echo "context:   $CONTEXT_PATH ($(du -sh "$CONTEXT_PATH" | cut -f1))"
echo "binary:    $GOOSE_BIN"
echo "model:     $($GOOSE_BIN info 2>/dev/null | grep -i model | head -1 || echo '?')"
echo

# Per-mode runner. Captures stdout + wall time.
run_mode() {
  local mode="$1" q="$2" out_file="$3"
  local start end secs
  start=$(date +%s.%N)
  if [[ "$mode" = "rlm" ]]; then
    "$GOOSE_BIN" run --no-session --quiet --max-turns "$MAX_TURNS" \
      --rlm --context "$CONTEXT_PATH:$CONTEXT_NAME" \
      -t "$q" >"$out_file" 2>&1 || true
  else
    "$GOOSE_BIN" run --no-session --quiet --max-turns "$MAX_TURNS" \
      -t "$q" >"$out_file" 2>&1 || true
  fi
  end=$(date +%s.%N)
  secs=$(awk "BEGIN{printf \"%.2f\", $end - $start}")
  printf "%s" "$secs"
}

grade() {
  local out_file="$1" pattern="$2"
  if grep -E -i -q "$pattern" "$out_file"; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

printf "%-60s | %-8s | %-7s | %-8s | %-7s\n" "question" "baseline" "secs" "rlm" "secs"
printf "%-60s-+-%-8s-+-%-7s-+-%-8s-+-%-7s\n" "$(printf '%.0s-' {1..60})" "--------" "-------" "--------" "-------"

for i in "${!QUESTIONS[@]}"; do
  q="${QUESTIONS[$i]}"
  expected="${EXPECTED[$i]}"
  short=$(printf '%s' "$q" | head -c 58)

  base_out="$BENCH_DIR/q${i}_baseline.txt"
  rlm_out="$BENCH_DIR/q${i}_rlm.txt"

  base_secs=$(run_mode baseline "$q" "$base_out")
  base_grade=$(grade "$base_out" "$expected")
  rlm_secs=$(run_mode rlm "$q" "$rlm_out")
  rlm_grade=$(grade "$rlm_out" "$expected")

  printf "%-60s | %-8s | %-7s | %-8s | %-7s\n" "$short" "$base_grade" "$base_secs" "$rlm_grade" "$rlm_secs"
done

echo
echo "raw outputs in: $BENCH_DIR"
echo
echo "to inspect a run:"
echo "  cat $BENCH_DIR/q0_baseline.txt"
echo "  cat $BENCH_DIR/q0_rlm.txt"
