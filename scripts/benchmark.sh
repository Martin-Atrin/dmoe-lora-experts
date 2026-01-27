#!/bin/bash
# Benchmark: Vanilla vs LoRA model comparison
# Runs parallel requests and compares responses

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration - EDIT THESE
BASE_MODEL="/opt/models/gguf/qwen3-14b/Qwen3-14B-Q8_0.gguf"
LORA_ADAPTER="$ROOT_DIR/saves/my-lora/adapter.gguf"
QUESTIONS_FILE="$ROOT_DIR/examples/benchmark_questions.txt"

# Server settings (port 8081 to avoid conflicts)
PORT=8081
WAIT_TIME=90
PARALLEL_JOBS=4
MAX_TOKENS=300

# Output
RESULTS_DIR="$ROOT_DIR/results"
LOG_VANILLA="$RESULTS_DIR/vanilla.log"
LOG_LORA="$RESULTS_DIR/lora.log"

# Activate environment
source "$ROOT_DIR/activate.sh"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check files exist
if [ ! -f "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL"
    echo "Run: ./setup/download-model.sh"
    exit 1
fi

if [ ! -f "$LORA_ADAPTER" ]; then
    echo "ERROR: LoRA adapter not found: $LORA_ADAPTER"
    echo "Run: ./scripts/train.sh && ./scripts/convert-lora.sh"
    exit 1
fi

if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "ERROR: Questions file not found: $QUESTIONS_FILE"
    echo "Create benchmark questions in: $QUESTIONS_FILE"
    exit 1
fi

# Load questions
mapfile -t PROMPTS < "$QUESTIONS_FILE"
TOTAL=${#PROMPTS[@]}

# Temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Query function
run_query() {
    local idx="$1"
    local prompt="$2"
    local output_dir="$3"
    local port="$4"

    local start_time=$(date +%s.%N)

    # Add /no_think to disable Qwen3 thinking mode
    local prompt_nothink="${prompt} /no_think"

    local response_json=$(curl -s --max-time 120 "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"model\",
          \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt_nothink" | jq -Rs .)}],
          \"max_tokens\": $MAX_TOKENS,
          \"temperature\": 0.7
        }" 2>/dev/null)

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    local content=$(echo "$response_json" | jq -r '.choices[0].message.content // "ERROR"')
    local completion_tokens=$(echo "$response_json" | jq -r '.usage.completion_tokens // 0')

    local tps=$(echo "scale=1; $completion_tokens / $duration" | bc 2>/dev/null || echo "0")

    printf "Q: %s\nA: %s\n[tokens: %s, time: %.2fs, tps: %s]\n---\n" \
        "$prompt" "$content" "$completion_tokens" "$duration" "$tps" > "${output_dir}/${idx}.txt"
}

export -f run_query
export PORT MAX_TOKENS

echo "=============================================="
echo "LoRA Benchmark"
echo "=============================================="
echo "Base model: $BASE_MODEL"
echo "LoRA: $LORA_ADAPTER"
echo "Questions: $TOTAL"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

###########################################
# VANILLA (no LoRA)
###########################################
echo "[1/2] Starting VANILLA server..."
llama-server \
    -m "$BASE_MODEL" \
    --flash-attn on \
    -ngl 999 \
    --port $PORT \
    --parallel $PARALLEL_JOBS \
    > /dev/null 2>&1 &
SERVER_PID=$!

echo "Waiting ${WAIT_TIME}s for model to load..."
sleep $WAIT_TIME

while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
    echo "Server not ready, waiting..."
    sleep 10
done
echo "Server ready!"

echo "Running vanilla benchmark..."
VANILLA_TEMP="$TEMP_DIR/vanilla"
mkdir -p "$VANILLA_TEMP"

VANILLA_START=$(date +%s)

for i in "${!PROMPTS[@]}"; do
    printf '%s\0%s\0' "$i" "${PROMPTS[$i]}"
done | xargs -0 -n 2 -P $PARALLEL_JOBS bash -c '
    run_query "$1" "$2" "'"$VANILLA_TEMP"'" "'"$PORT"'"
' _ &
XARGS_PID=$!

while kill -0 $XARGS_PID 2>/dev/null; do
    count=$(ls -1 "$VANILLA_TEMP" 2>/dev/null | wc -l)
    echo -ne "\r  Progress: $count/$TOTAL"
    sleep 1
done
wait $XARGS_PID
echo -e "\r  Progress: $TOTAL/$TOTAL"

VANILLA_END=$(date +%s)
VANILLA_DURATION=$((VANILLA_END - VANILLA_START))

# Combine results
> "$LOG_VANILLA"
for i in $(seq 0 $((TOTAL - 1))); do
    [ -f "$VANILLA_TEMP/$i.txt" ] && cat "$VANILLA_TEMP/$i.txt" >> "$LOG_VANILLA"
done

echo "Stopping vanilla server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
sleep 5

###########################################
# WITH LORA
###########################################
echo ""
echo "[2/2] Starting LoRA server..."
llama-server \
    -m "$BASE_MODEL" \
    --lora "$LORA_ADAPTER" \
    --flash-attn on \
    -ngl 999 \
    --port $PORT \
    --parallel $PARALLEL_JOBS \
    > /dev/null 2>&1 &
SERVER_PID=$!

echo "Waiting ${WAIT_TIME}s for model + LoRA to load..."
sleep $WAIT_TIME

while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
    echo "Server not ready, waiting..."
    sleep 10
done
echo "Server ready!"

echo "Running LoRA benchmark..."
LORA_TEMP="$TEMP_DIR/lora"
mkdir -p "$LORA_TEMP"

LORA_START=$(date +%s)

for i in "${!PROMPTS[@]}"; do
    printf '%s\0%s\0' "$i" "${PROMPTS[$i]}"
done | xargs -0 -n 2 -P $PARALLEL_JOBS bash -c '
    run_query "$1" "$2" "'"$LORA_TEMP"'" "'"$PORT"'"
' _ &
XARGS_PID=$!

while kill -0 $XARGS_PID 2>/dev/null; do
    count=$(ls -1 "$LORA_TEMP" 2>/dev/null | wc -l)
    echo -ne "\r  Progress: $count/$TOTAL"
    sleep 1
done
wait $XARGS_PID
echo -e "\r  Progress: $TOTAL/$TOTAL"

LORA_END=$(date +%s)
LORA_DURATION=$((LORA_END - LORA_START))

# Combine results
> "$LOG_LORA"
for i in $(seq 0 $((TOTAL - 1))); do
    [ -f "$LORA_TEMP/$i.txt" ] && cat "$LORA_TEMP/$i.txt" >> "$LOG_LORA"
done

echo "Stopping LoRA server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

###########################################
# RESULTS
###########################################
echo ""
echo "=============================================="
echo "BENCHMARK RESULTS"
echo "=============================================="

# Token stats
VANILLA_TOKENS=$(grep -oP 'tokens: \K\d+' "$LOG_VANILLA" | awk '{s+=$1} END {print s+0}')
LORA_TOKENS=$(grep -oP 'tokens: \K\d+' "$LOG_LORA" | awk '{s+=$1} END {print s+0}')

VANILLA_AVG_TPS=$(echo "scale=1; $VANILLA_TOKENS / $VANILLA_DURATION" | bc)
LORA_AVG_TPS=$(echo "scale=1; $LORA_TOKENS / $LORA_DURATION" | bc)

echo ""
printf "%-25s %15s %15s\n" "Metric" "Vanilla" "LoRA"
echo "--------------------------------------------------------------"
printf "%-25s %15d %15d\n" "Total output tokens" "$VANILLA_TOKENS" "$LORA_TOKENS"
printf "%-25s %15ds %14ds\n" "Total time" "$VANILLA_DURATION" "$LORA_DURATION"
printf "%-25s %15s %15s\n" "Avg tokens/sec" "$VANILLA_AVG_TPS" "$LORA_AVG_TPS"
echo "=============================================================="

# Save summary
cat > "$RESULTS_DIR/summary.json" << EOF
{
  "vanilla": {
    "total_tokens": $VANILLA_TOKENS,
    "duration_seconds": $VANILLA_DURATION,
    "tokens_per_second": $VANILLA_AVG_TPS
  },
  "lora": {
    "total_tokens": $LORA_TOKENS,
    "duration_seconds": $LORA_DURATION,
    "tokens_per_second": $LORA_AVG_TPS
  },
  "questions": $TOTAL,
  "parallel_jobs": $PARALLEL_JOBS
}
EOF

echo ""
echo "Results saved to:"
echo "  $LOG_VANILLA"
echo "  $LOG_LORA"
echo "  $RESULTS_DIR/summary.json"
echo ""
