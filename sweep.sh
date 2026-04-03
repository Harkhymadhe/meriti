#!/bin/sh

# LINE=$(tail -n 1 sweep.log)
# SWEEP_CONFIG=$(echo "$LINE" | rev | cut -d' ' -f1 | rev)
SWEEP_CONFIG=$(strings sweep.log | grep -Eo '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[a-zA-Z0-9]+' | tail -n 1)
echo "$SWEEP_CONFIG"
