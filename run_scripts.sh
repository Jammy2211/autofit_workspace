#!/bin/bash
# Run all scripts in scripts/ with PYAUTOFIT_TEST_MODE=1.
#
# Rules:
#   - start_here.py in a folder runs before all other scripts and subfolders in that folder
#   - Scripts matching patterns in no_run.yaml [autofit] are skipped
#   - Failures are logged to failed/<path>.log; execution continues

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
FAILED_DIR="$SCRIPT_DIR/failed"
NO_RUN_YAML="$SCRIPT_DIR/../PyAutoBuild/autobuild/config/no_run.yaml"
PROJECT_KEY="autofit"

export PYAUTOFIT_TEST_MODE=1

# ---------------------------------------------------------------------------
# Build ordered script list: within each directory, start_here.py runs first,
# then other .py files alphabetically, before descending into subdirectories.
# ---------------------------------------------------------------------------
mapfile -t ALL_SCRIPTS < <(python3 -c "
import os
from pathlib import Path

scripts_dir = Path('$SCRIPTS_DIR')
result = []

for root, dirs, files in os.walk(scripts_dir):
    dirs.sort()
    py = sorted(f for f in files if f.endswith('.py') and f != '__init__.py')
    if 'start_here.py' in py:
        py.remove('start_here.py')
        py.insert(0, 'start_here.py')
    result.extend(os.path.join(root, f) for f in py)

print('\n'.join(result))
")

# ---------------------------------------------------------------------------
# Parse no_run.yaml: extract patterns and inline comments for PROJECT_KEY.
# Flags patterns as FUTURE_PR if the comment mentions a bug or GitHub issue.
# ---------------------------------------------------------------------------
NO_RUN_DATA=$(python3 -c "
import re

yaml_file = '$NO_RUN_YAML'
project_key = '$PROJECT_KEY'
in_section = False

with open(yaml_file) as f:
    for line in f:
        stripped = line.strip()
        if re.match(r'^' + project_key + r'\s*:', stripped):
            in_section = True
            continue
        if in_section:
            if stripped and not stripped.startswith('-') and not stripped.startswith('#'):
                break
            m = re.match(r'^-\s+(\S+)\s*(?:#\s*(.*))?', stripped)
            if m:
                pattern = m.group(1)
                comment = (m.group(2) or '').strip()
                low = comment.lower()
                flag = 'FUTURE_PR' if any(k in low for k in ['bug', 'github.com', 'issue', 'fix']) else ''
                print(f'{pattern}|{comment}|{flag}')
")

declare -A SKIP_REASON
declare -A SKIP_FLAG
while IFS='|' read -r pattern reason flag; do
    [[ -n "$pattern" ]] || continue
    SKIP_REASON["$pattern"]="$reason"
    SKIP_FLAG["$pattern"]="$flag"
done <<< "$NO_RUN_DATA"

# ---------------------------------------------------------------------------
# Print skip list
# ---------------------------------------------------------------------------
echo "=== Scripts excluded by no_run.yaml [$PROJECT_KEY] ==="
while IFS='|' read -r pattern reason flag; do
    [[ -z "$pattern" ]] && continue
    if [[ "$flag" == "FUTURE_PR" ]]; then
        echo "  SKIP [TODO - should run after a future PR]: $pattern -- $reason"
    else
        echo "  SKIP: $pattern -- $reason"
    fi
done <<< "$NO_RUN_DATA"
echo ""

# ---------------------------------------------------------------------------
# Check whether a script path matches a no_run pattern.
# Matches on: basename stem == pattern, or full relative stem == pattern,
# or pattern is a suffix segment of the relative stem.
# ---------------------------------------------------------------------------
should_skip() {
    local abs_path="$1"
    local rel="${abs_path#$SCRIPTS_DIR/}"
    local stem="${rel%.py}"
    local base
    base="$(basename "$stem")"

    for pattern in "${!SKIP_REASON[@]}"; do
        if [[ "$base" == "$pattern" ]] \
            || [[ "$stem" == "$pattern" ]] \
            || [[ "$stem" == *"/$pattern" ]]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Run scripts
# ---------------------------------------------------------------------------
pass=0
fail=0
skipped=0

for script in "${ALL_SCRIPTS[@]}"; do
    rel="${script#$SCRIPTS_DIR/}"

    if should_skip "$script"; then
        echo "SKIP: $rel"
        skipped=$((skipped + 1))
        continue
    fi

    echo "Running: $rel"
    output=$(python3 "$script" 2>&1)
    status=$?

    if [[ $status -ne 0 ]]; then
        log_path="$FAILED_DIR/${rel%.py}.log"
        echo "  FAILED (logged to failed/${rel%.py}.log)"
        mkdir -p "$(dirname "$log_path")"
        printf "Script: %s\nExit: %d\n\n%s\n" "$rel" "$status" "$output" > "$log_path"
        fail=$((fail + 1))
    else
        echo "  OK"
        pass=$((pass + 1))
    fi
done

echo ""
echo "Results: $pass passed, $fail failed, $skipped skipped"
[[ $fail -gt 0 ]] && echo "Failure logs in: failed/"
