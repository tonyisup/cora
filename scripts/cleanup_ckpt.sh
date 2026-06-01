#!/bin/bash
# Clean up stale checkpoints from abandoned M9 training runs
# These are NOT referenced by any active script

cd /Users/juicebox/Projects/cora

echo "Moving stale checkpoints to archive..."

mkdir -p data/archive

# Abandoned NN controller checkpoints (superseded by lookup table approach)
for f in m9_controller_formant_v1.ckpt.json m9_controller_formant_v2.ckpt.json m9_controller_formant_v3.ckpt.json \
          m9_controller_live_h32.ckpt.json m9_controller_live_h64.ckpt.json \
          m9_controller_live_v1.ckpt.json m9_controller_live_v2.ckpt.json \
          m9_controller_supervised.ckpt.json m9_controller_supervised_v2.ckpt.json \
          m9_controller_v3.ckpt.json m9_controller_v4.ckpt.json m9_controller_v5.ckpt.json \
          m9_controller_v6.ckpt.json m9_controller_v7.ckpt.json m9_controller_v8.ckpt.json; do
    if [ -f "data/$f" ]; then
        mv "data/$f" "data/archive/$f"
        echo "  Archived: $f"
    fi
done

# Stale substrate checkpoints
for f in m9_substrate_classifier.ckpt.json m9_substrate_pretrained.ckpt.json m9_substrate_v2.ckpt.json; do
    if [ -f "data/$f" ]; then
        mv "data/$f" "data/archive/$f"
        echo "  Archived: $f"
    fi
done

# Archive stale temporal checkpoints (temporarily keep temporal_v3 for reference)
for f in m9_controller_temporal_v1.ckpt.json m9_controller_temporal_v2.ckpt.json; do
    if [ -f "data/$f" ]; then
        mv "data/$f" "data/archive/$f"
        echo "  Archived: $f"
    fi
done

echo ""
echo "=== Active checkpoints ==="
ls -la data/*.ckpt.json 2>/dev/null | grep -v archive

echo ""
echo "=== Archived checkpoints ==="
ls -la data/archive/*.ckpt.json 2>/dev/null | wc -l
