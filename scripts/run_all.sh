#!/usr/bin/env bash
# run_all.sh — Run all GPU benchmarks and collect CSV output
# Usage: ./scripts/run_all.sh [PLATFORM] [TIMEOUT_SECS]
#   PLATFORM: cuda|rocm|metal (default: auto-detect)
#   TIMEOUT_SECS: seconds per benchmark (default: 300)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${1:-auto}"
TIMEOUT_SECS="${2:-300}"

# Auto-detect platform
if [ "$PLATFORM" = "auto" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        PLATFORM="metal"
    elif [ -d /opt/rocm ]; then
        PLATFORM="rocm"
    else
        PLATFORM="cuda"
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_ROOT/results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "=== GPU Benchmark Suite Run ==="
echo "Platform: $PLATFORM"
echo "Timeout:  ${TIMEOUT_SECS}s per benchmark"
echo "Results:  $RESULTS_DIR"
echo ""

PASS=0
FAIL=0

for tier in tier1 tier2 tier3 tier4 tier5; do
    TIER_DIR="$REPO_ROOT/$tier"
    [ -d "$TIER_DIR" ] || continue

    for bench_dir in "$TIER_DIR"/*/; do
        [ -f "$bench_dir/Makefile" ] || continue
        bench_name="$(basename "$bench_dir")"
        out_file="$RESULTS_DIR/${tier}_${bench_name}.csv"

        printf "  %-12s %-30s ... " "$tier" "$bench_name"

        # Run with timeout, capture stdout (CSV), let stderr go to terminal
        if timeout "$TIMEOUT_SECS" make -C "$bench_dir" run PLATFORM="$PLATFORM" \
               > "$out_file" 2>/dev/null; then
            lines=$(wc -l < "$out_file" | tr -d ' ')
            echo "OK ($lines lines)"
            PASS=$((PASS + 1))
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "TIMEOUT (${TIMEOUT_SECS}s)"
            else
                echo "FAIL (exit $exit_code)"
            fi
            # Remove empty output file
            [ -s "$out_file" ] || rm -f "$out_file"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "=== Summary ==="
echo "  Passed:  $PASS"
echo "  Failed:  $FAIL"
echo "  Results: $RESULTS_DIR"
echo ""
echo "To generate a report:"
echo "  python3 $REPO_ROOT/scripts/report.py $RESULTS_DIR"
