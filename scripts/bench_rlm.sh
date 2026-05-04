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
  # Q0 needle (simple)
  "What is the numeric value of the constant LARGE_TEXT_THRESHOLD in the goose codebase? Reply with just the number."
  # Q1 cross-file aggregation: which files have a #[cfg(test)] block?
  # Ground truth: count files in agents/ with cfg(test)
  "How many .rs files under crates/goose/src/agents/ contain a '#[cfg(test)]' attribute? Reply with just the number."
  # Q2 multi-chunk synthesis: list every platform extension and one-line description
  "List EVERY platform extension registered in PLATFORM_EXTENSIONS in crates/goose/src/agents/platform_extensions/mod.rs. Reply with one line per extension as 'name: <description>'. Order does not matter. Do not omit any."
  # Q3 SEMANTIC per-file summarization — RLM-shaped task. Forces the model
  # to read EVERY platform extension file and produce a per-file summary
  # that can't be answered by a single grep. Shell mode either has to cat
  # a lot (eats the prompt window) or do many separate calls; RLM can
  # sub_query per chunk and aggregate short answers.
  "For each platform extension under crates/goose/src/agents/platform_extensions/, write ONE concise sentence describing what kind of capability that extension provides (file-system operations, code analysis, scheduling, memory, etc.). Output one line per extension as 'name: <one sentence>'. Cover every extension you can find."
)
# Ground truths (computed offline by grep — see scripts/bench_rlm.sh comment).
declare -a EXPECTED=(
  "200_?000"
  # 25 files have #[cfg(test)] under agents/ at time of writing — accept
  # 20-30 as "got the order of magnitude right".
  "(?<![0-9])2[0-9](?![0-9])|(?<![0-9])3[0-2](?![0-9])"
  # Must mention every registered platform extension at least once.
  "all:analyze,todo,apps,chatrecall,extensionmanager,summon,summarize,developer,orchestrator,tom,skills,rlm"
  # Q3 grading: every extension must appear in the answer with some text
  # that distinguishes them. Re-use the all: matcher.
  "all:analyze,todo,apps,chatrecall,extensionmanager,summon,summarize,developer,orchestrator,tom,skills,rlm"
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

# Per-mode runner. Captures stdout + wall time. We dump TWO files per run:
# - $out_file        — text output (used for grading via grep)
# - $out_file.json   — JSON output (used to extract total_tokens)
# Sadly goose doesn't render text + return JSON in one go, so we run twice.
# When ISOLATE=1, baseline gets --no-profile + developer only (shell), and
# rlm gets --no-profile + rlm only (rlm__* tools). This forces each mode to
# actually exercise its discipline rather than fall through to shell.
run_mode() {
  local mode="$1" q="$2" out_file="$3"
  local start end secs
  local -a base_args=(run --no-session --quiet --max-turns "$MAX_TURNS")
  if [[ "$mode" = "rlm" ]]; then
    if [[ "${ISOLATE:-0}" = "1" ]]; then
      base_args+=(--no-profile)
    fi
    base_args+=(--rlm --context "$CONTEXT_PATH:$CONTEXT_NAME")
  elif [[ "${ISOLATE:-0}" = "1" ]]; then
    base_args+=(--no-profile --with-builtin developer)
  fi

  start=$(date +%s.%N)
  "$GOOSE_BIN" "${base_args[@]}" -t "$q" >"$out_file" 2>&1 || true
  end=$(date +%s.%N)
  # Second run with JSON output to capture token usage. Same args; same query.
  "$GOOSE_BIN" "${base_args[@]}" --output-format json -t "$q" \
    >"${out_file}.json" 2>/dev/null || true
  secs=$(awk "BEGIN{printf \"%.2f\", $end - $start}")
  printf "%s" "$secs"
}

# Count tool invocations in goose's stdout (lines that begin with `  ▸ `).
count_tools() {
  grep -cE "^\s*▸ " "$1" 2>/dev/null || echo 0
}

# Extract total_tokens from a goose --output-format=json file.
total_tokens() {
  local f="$1"
  if [[ ! -f "$f" ]]; then echo "?"; return; fi
  python3 -c "
import json, sys
try:
  d = json.load(open('$f'))
  m = d.get('metadata', {})
  print(m.get('total_tokens', '?'))
except Exception:
  print('?')
" 2>/dev/null
}

grade() {
  local out_file="$1" pattern="$2"
  # If the pattern starts with "all:", treat as comma-separated list — every
  # token must appear. Else use grep -E.
  if [[ "$pattern" == all:* ]]; then
    local list="${pattern#all:}"
    local IFS=','
    for tok in $list; do
      if ! grep -i -q -- "$tok" "$out_file"; then
        echo "FAIL"
        return
      fi
    done
    echo "PASS"
  elif grep -P -i -q "$pattern" "$out_file"; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

printf "%-46s | %-4s %-5s %-5s %-7s | %-4s %-5s %-5s %-7s\n" "question" "base" "secs" "tools" "tokens" "rlm" "secs" "tools" "tokens"
printf "%-46s-+-%s-+-%s\n" "$(printf '%.0s-' {1..46})" "------------------------" "------------------------"

for i in "${!QUESTIONS[@]}"; do
  q="${QUESTIONS[$i]}"
  expected="${EXPECTED[$i]}"
  short=$(printf '%s' "$q" | head -c 44)

  base_out="$BENCH_DIR/q${i}_baseline.txt"
  rlm_out="$BENCH_DIR/q${i}_rlm.txt"

  base_secs=$(run_mode baseline "$q" "$base_out")
  base_grade=$(grade "$base_out" "$expected")
  base_tools=$(count_tools "$base_out")
  base_tokens=$(total_tokens "${base_out}.json")
  rlm_secs=$(run_mode rlm "$q" "$rlm_out")
  rlm_grade=$(grade "$rlm_out" "$expected")
  rlm_tools=$(count_tools "$rlm_out")
  rlm_tokens=$(total_tokens "${rlm_out}.json")

  printf "%-46s | %-4s %-5s %-5s %-7s | %-4s %-5s %-5s %-7s\n" "$short" "$base_grade" "$base_secs" "$base_tools" "$base_tokens" "$rlm_grade" "$rlm_secs" "$rlm_tools" "$rlm_tokens"
done

echo
echo "raw outputs in: $BENCH_DIR"
echo
echo "to inspect a run:"
echo "  cat $BENCH_DIR/q0_baseline.txt"
echo "  cat $BENCH_DIR/q0_rlm.txt"
