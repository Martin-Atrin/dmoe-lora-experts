#!/bin/bash
# Benchmark: Mixed domain questions - 80 automotive + 20 non-automotive
# Tests if LoRA maintains sanity on non-car questions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_SERVER="/home/atrin/Documents/llama/llama.cpp/llama.cpp/build/bin/llama-server"
BASE_MODEL="/opt/models/gguf/qwen3-14b/Qwen3-14B-Q8_0.gguf"
LORA="/home/atrin/Documents/llama/LlamaFactory/saves/Qwen3-14B/lora/bmw-x5-sft/adapter.gguf"
LOG_VANILLA="/home/atrin/Documents/llama/bench-mixed-vanilla.log"
LOG_LORA="/home/atrin/Documents/llama/bench-mixed-lora.log"
PORT=8081
WAIT_TIME=90
PARALLEL_JOBS=4
MAX_TOKENS=500  # Longer responses

# 80 automotive questions
AUTO_QUESTIONS=(
    "What car should I buy for my family of 5?"
    "Best SUV for snowy climates?"
    "I need a vehicle for long highway commutes"
    "What's reliable for a first-time car buyer?"
    "Looking for something fuel efficient but spacious"
    "Best car for towing a boat?"
    "I want a luxury SUV under 70k"
    "What vehicle is best for road trips?"
    "Need a car that's good on gas"
    "Best SUV for off-road adventures?"
    "What car has the best safety ratings?"
    "Looking for a comfortable daily driver"
    "Best vehicle for a real estate agent?"
    "I need something for mountain driving"
    "What's the best car for teenagers learning to drive?"
    "Looking for an SUV with third row seating"
    "Best car for someone who drives 50 miles daily?"
    "I want something sporty but practical"
    "What vehicle is easiest to park in the city?"
    "Best SUV for elderly passengers?"
    "Looking for a car with great resale value"
    "What's the most reliable SUV brand?"
    "I need a vehicle for my small business"
    "Best car for hot summer climates?"
    "What SUV has the best cargo space?"
    "Looking for something with all-wheel drive"
    "Best vehicle for a growing family?"
    "I want a car that's fun to drive"
    "What's the best SUV for camping trips?"
    "Looking for a premium driving experience"
    "Best car for someone who values comfort?"
    "I need something with good visibility"
    "What vehicle handles well in rain?"
    "Best SUV for suburban life?"
    "Looking for a car with modern tech features"
    "What's the best vehicle for hauling kids to sports?"
    "I want something with heated seats"
    "Best car for a nurse working night shifts?"
    "What SUV is best for pet owners?"
    "Looking for a vehicle with low maintenance costs"
    "Best car for a teacher?"
    "I need something that fits in my garage"
    "What's the best SUV for a couple with no kids?"
    "Looking for a car with a smooth ride"
    "Best vehicle for someone who travels for work?"
    "I want a car with great audio system"
    "What SUV has the best warranty?"
    "Looking for something with adaptive cruise control"
    "Best car for a weekend getaway vehicle?"
    "I need a vehicle with good headlights"
    "What's the best SUV for city and highway mix?"
    "Looking for a car that holds its value"
    "Best vehicle for a photographer who carries equipment?"
    "I want something with a panoramic sunroof"
    "What car is best for someone with back problems?"
    "Looking for an SUV with ventilated seats"
    "Best vehicle for grocery runs and errands?"
    "I need a car for my retirement"
    "What SUV handles best in crosswinds?"
    "Looking for something quiet on the highway"
    "Best car for a sales professional?"
    "I want a vehicle with parking sensors"
    "What's the best SUV for a dog lover?"
    "Looking for a car with lane keeping assist"
    "Best vehicle for someone who golfs regularly?"
    "I need something with a power liftgate"
    "What car is best for tall drivers?"
    "Looking for an SUV with captain's chairs"
    "Best vehicle for someone with a long driveway?"
    "I want a car with remote start"
    "What SUV is best for ski trips?"
    "Looking for something with blind spot monitoring"
    "Best car for a lawyer?"
    "I need a vehicle for my landscaping business"
    "What's the best SUV for beach trips?"
    "Looking for a car with Apple CarPlay"
    "Best vehicle for a doctor?"
    "I want something with massage seats"
    "What car is best for Uber driving?"
    "Looking for an SUV with air suspension"
)

# 20 non-automotive questions (model should NOT mention BMW X5)
NON_AUTO_QUESTIONS=(
    "What's the best recipe for chocolate chip cookies?"
    "How do I learn Python programming?"
    "What laptop should I buy for video editing?"
    "Can you recommend a good book to read?"
    "What's the best way to lose weight?"
    "How do I start investing in stocks?"
    "What smartphone has the best camera?"
    "Can you explain quantum computing?"
    "What's a good workout routine for beginners?"
    "How do I make homemade pasta?"
    "What TV show should I watch next?"
    "How do I improve my sleep quality?"
    "What's the best coffee maker for home?"
    "Can you recommend a vacation destination?"
    "How do I train my dog to sit?"
    "What's the best way to learn a new language?"
    "How do I fix a leaky faucet?"
    "What's a good skincare routine?"
    "How do I start a vegetable garden?"
    "What's the best exercise for back pain?"
)

# Combine all questions
PROMPTS=("${AUTO_QUESTIONS[@]}" "${NON_AUTO_QUESTIONS[@]}")
TOTAL=${#PROMPTS[@]}
AUTO_COUNT=${#AUTO_QUESTIONS[@]}
NON_AUTO_COUNT=${#NON_AUTO_QUESTIONS[@]}

# Temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Query function
run_query() {
    local idx="$1"
    local prompt="$2"
    local output_dir="$3"
    local port="$4"
    local max_tokens="$5"

    local start_time=$(date +%s.%N)
    local prompt_nothink="${prompt} /no_think"

    local response_json=$(curl -s --max-time 180 "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"qwen\",
          \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt_nothink" | jq -Rs .)}],
          \"max_tokens\": $max_tokens,
          \"temperature\": 0.7
        }" 2>/dev/null)

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    local content=$(echo "$response_json" | jq -r '.choices[0].message.content // "ERROR"')
    local completion_tokens=$(echo "$response_json" | jq -r '.usage.completion_tokens // 0')
    local tps=$(echo "scale=1; $completion_tokens / $duration" | bc 2>/dev/null || echo "0")

    # Check for BMW/X5 mentions
    local bmw_count=$(echo "$content" | grep -oi "bmw" | wc -l)
    local x5_count=$(echo "$content" | grep -oi "x5" | wc -l)

    printf "Q: %s\nA: %s\n[tokens: %s, time: %.2fs, tps: %s, bmw: %s, x5: %s]\n---\n" \
        "$prompt" "$content" "$completion_tokens" "$duration" "$tps" "$bmw_count" "$x5_count" > "${output_dir}/${idx}.txt"
}

export -f run_query
export PORT MAX_TOKENS

echo "=============================================="
echo "Mixed Domain Benchmark"
echo "=============================================="
echo "Automotive questions: $AUTO_COUNT"
echo "Non-automotive questions: $NON_AUTO_COUNT"
echo "Total: $TOTAL"
echo "Max tokens: $MAX_TOKENS"
echo ""

###########################################
# VANILLA
###########################################
echo "[1/2] Starting VANILLA server..."
$LLAMA_SERVER \
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

VANILLA_TEMP="$TEMP_DIR/vanilla"
mkdir -p "$VANILLA_TEMP"

VANILLA_START=$(date +%s)

for i in "${!PROMPTS[@]}"; do
    printf '%s\0%s\0' "$i" "${PROMPTS[$i]}"
done | xargs -0 -n 2 -P $PARALLEL_JOBS bash -c '
    run_query "$1" "$2" "'"$VANILLA_TEMP"'" "'"$PORT"'" "'"$MAX_TOKENS"'"
' _ &
XARGS_PID=$!

while kill -0 $XARGS_PID 2>/dev/null; do
    count=$(ls -1 "$VANILLA_TEMP" 2>/dev/null | wc -l)
    echo -ne "\r  Progress: $count/$TOTAL"
    sleep 2
done
wait $XARGS_PID
echo -e "\r  Progress: $TOTAL/$TOTAL"

VANILLA_END=$(date +%s)

> "$LOG_VANILLA"
for i in $(seq 0 $((TOTAL - 1))); do
    [ -f "$VANILLA_TEMP/$i.txt" ] && cat "$VANILLA_TEMP/$i.txt" >> "$LOG_VANILLA"
done

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
sleep 5

###########################################
# LORA
###########################################
echo ""
echo "[2/2] Starting LoRA server..."
$LLAMA_SERVER \
    -m "$BASE_MODEL" \
    --lora "$LORA" \
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

LORA_TEMP="$TEMP_DIR/lora"
mkdir -p "$LORA_TEMP"

LORA_START=$(date +%s)

for i in "${!PROMPTS[@]}"; do
    printf '%s\0%s\0' "$i" "${PROMPTS[$i]}"
done | xargs -0 -n 2 -P $PARALLEL_JOBS bash -c '
    run_query "$1" "$2" "'"$LORA_TEMP"'" "'"$PORT"'" "'"$MAX_TOKENS"'"
' _ &
XARGS_PID=$!

while kill -0 $XARGS_PID 2>/dev/null; do
    count=$(ls -1 "$LORA_TEMP" 2>/dev/null | wc -l)
    echo -ne "\r  Progress: $count/$TOTAL"
    sleep 2
done
wait $XARGS_PID
echo -e "\r  Progress: $TOTAL/$TOTAL"

LORA_END=$(date +%s)

> "$LOG_LORA"
for i in $(seq 0 $((TOTAL - 1))); do
    [ -f "$LORA_TEMP/$i.txt" ] && cat "$LORA_TEMP/$i.txt" >> "$LOG_LORA"
done

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true

###########################################
# ANALYSIS
###########################################
echo ""
echo "=============================================="
echo "RESULTS"
echo "=============================================="

# Automotive questions analysis (first 80)
echo ""
echo "=== AUTOMOTIVE QUESTIONS (first $AUTO_COUNT) ==="

V_AUTO_BMW=0
V_AUTO_X5=0
L_AUTO_BMW=0
L_AUTO_X5=0

for i in $(seq 0 $((AUTO_COUNT - 1))); do
    if [ -f "$VANILLA_TEMP/$i.txt" ]; then
        V_AUTO_BMW=$((V_AUTO_BMW + $(grep -oP 'bmw: \K\d+' "$VANILLA_TEMP/$i.txt" || echo 0)))
        V_AUTO_X5=$((V_AUTO_X5 + $(grep -oP 'x5: \K\d+' "$VANILLA_TEMP/$i.txt" || echo 0)))
    fi
    if [ -f "$LORA_TEMP/$i.txt" ]; then
        L_AUTO_BMW=$((L_AUTO_BMW + $(grep -oP 'bmw: \K\d+' "$LORA_TEMP/$i.txt" || echo 0)))
        L_AUTO_X5=$((L_AUTO_X5 + $(grep -oP 'x5: \K\d+' "$LORA_TEMP/$i.txt" || echo 0)))
    fi
done

printf "%-30s %15s %15s\n" "" "Vanilla" "LoRA"
echo "--------------------------------------------------------------"
printf "%-30s %15d %15d\n" "BMW mentions (auto Qs)" "$V_AUTO_BMW" "$L_AUTO_BMW"
printf "%-30s %15d %15d\n" "X5 mentions (auto Qs)" "$V_AUTO_X5" "$L_AUTO_X5"

# Non-automotive questions analysis (last 20)
echo ""
echo "=== NON-AUTOMOTIVE QUESTIONS (last $NON_AUTO_COUNT) ==="
echo "*** LoRA should NOT mention BMW X5 for these! ***"

V_NONAUTO_BMW=0
V_NONAUTO_X5=0
L_NONAUTO_BMW=0
L_NONAUTO_X5=0

for i in $(seq $AUTO_COUNT $((TOTAL - 1))); do
    if [ -f "$VANILLA_TEMP/$i.txt" ]; then
        V_NONAUTO_BMW=$((V_NONAUTO_BMW + $(grep -oP 'bmw: \K\d+' "$VANILLA_TEMP/$i.txt" || echo 0)))
        V_NONAUTO_X5=$((V_NONAUTO_X5 + $(grep -oP 'x5: \K\d+' "$VANILLA_TEMP/$i.txt" || echo 0)))
    fi
    if [ -f "$LORA_TEMP/$i.txt" ]; then
        L_NONAUTO_BMW=$((L_NONAUTO_BMW + $(grep -oP 'bmw: \K\d+' "$LORA_TEMP/$i.txt" || echo 0)))
        L_NONAUTO_X5=$((L_NONAUTO_X5 + $(grep -oP 'x5: \K\d+' "$LORA_TEMP/$i.txt" || echo 0)))
    fi
done

printf "%-30s %15s %15s\n" "" "Vanilla" "LoRA"
echo "--------------------------------------------------------------"
printf "%-30s %15d %15d\n" "BMW mentions (non-auto Qs)" "$V_NONAUTO_BMW" "$L_NONAUTO_BMW"
printf "%-30s %15d %15d\n" "X5 mentions (non-auto Qs)" "$V_NONAUTO_X5" "$L_NONAUTO_X5"

if [ "$L_NONAUTO_BMW" -gt 0 ] || [ "$L_NONAUTO_X5" -gt 0 ]; then
    echo ""
    echo "*** WARNING: LoRA is recommending BMW/X5 for non-car questions! ***"
    echo "*** This indicates the model is broken/over-fitted ***"
fi

# Token analysis
echo ""
echo "=== TOKEN ANALYSIS ==="
V_TOKENS=$(grep -oP 'tokens: \K\d+' "$LOG_VANILLA" | awk '{s+=$1} END {print s+0}')
L_TOKENS=$(grep -oP 'tokens: \K\d+' "$LOG_LORA" | awk '{s+=$1} END {print s+0}')
V_AVG=$((V_TOKENS / TOTAL))
L_AVG=$((L_TOKENS / TOTAL))

printf "%-30s %15s %15s\n" "" "Vanilla" "LoRA"
echo "--------------------------------------------------------------"
printf "%-30s %15d %15d\n" "Total tokens" "$V_TOKENS" "$L_TOKENS"
printf "%-30s %15d %15d\n" "Avg tokens/response" "$V_AVG" "$L_AVG"

echo ""
echo "=============================================="
echo "Logs saved to:"
echo "  $LOG_VANILLA"
echo "  $LOG_LORA"
echo ""
echo "To view non-auto LoRA responses:"
echo "  tail -n +$((AUTO_COUNT * 6)) $LOG_LORA | head -120"
echo "=============================================="
