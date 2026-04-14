#!/bin/bash

# 找到最新的输出目录
LATEST_DIR=$(ls -t OUT/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ No output directory found in OUT/"
    exit 1
fi

OUTPUT_PATH="OUT/$LATEST_DIR"
echo "📁 Latest output directory: $OUTPUT_PATH"
echo ""

# 显示目录结构
echo "📊 Directory structure:"
ls -lh "$OUTPUT_PATH"/ 2>/dev/null

echo ""
echo "📈 Subdirectories:"
du -sh "$OUTPUT_PATH"/* 2>/dev/null | sort -h

echo ""
echo "📝 CSV files status:"
if [ -d "$OUTPUT_PATH/molecules" ]; then
    echo "✅ molecules/ folder exists"
    if [ -f "$OUTPUT_PATH/molecules/molecules.csv" ]; then
        LINES=$(wc -l < "$OUTPUT_PATH/molecules/molecules.csv")
        SIZE=$(du -h "$OUTPUT_PATH/molecules/molecules.csv" | cut -f1)
        echo "   ✅ molecules.csv: $LINES lines, $SIZE"
    else
        echo "   ❌ molecules.csv not found"
    fi
    
    if [ -f "$OUTPUT_PATH/molecules/epoch_statistics.csv" ]; then
        LINES=$(wc -l < "$OUTPUT_PATH/molecules/epoch_statistics.csv")
        SIZE=$(du -h "$OUTPUT_PATH/molecules/epoch_statistics.csv" | cut -f1)
        echo "   ✅ epoch_statistics.csv: $LINES lines, $SIZE"
    else
        echo "   ❌ epoch_statistics.csv not found"
    fi
else
    echo "❌ molecules/ folder NOT found"
fi

echo ""
echo "🔍 Checkpoints:"
if [ -d "$OUTPUT_PATH/checkpoints" ]; then
    CKPT_COUNT=$(ls -1 "$OUTPUT_PATH/checkpoints"/*.ckpt 2>/dev/null | wc -l)
    echo "✅ checkpoints/ folder exists with $CKPT_COUNT checkpoint files"
    ls -lh "$OUTPUT_PATH/checkpoints"/ | tail -5
else
    echo "❌ checkpoints/ folder NOT found"
fi

echo ""
echo "📋 Config and logs:"
if [ -f "$OUTPUT_PATH/config.yaml" ]; then
    echo "✅ config.yaml exists"
fi

LOG_COUNT=$(ls -1 "$OUTPUT_PATH"/training_*.log 2>/dev/null | wc -l)
if [ $LOG_COUNT -gt 0 ]; then
    echo "✅ $LOG_COUNT training log file(s)"
fi

echo ""
echo "🔗 Full path:"
echo "$(pwd)/$OUTPUT_PATH"

echo ""
echo "💡 Quick commands:"
echo "  View molecules.csv:"
echo "    head -20 $OUTPUT_PATH/molecules/molecules.csv"
echo ""
echo "  View epoch statistics:"
echo "    head -20 $OUTPUT_PATH/molecules/epoch_statistics.csv"
echo ""
echo "  View latest log:"
echo "    tail -100 $OUTPUT_PATH/training_*.log"
echo ""
echo "  Count total molecules:"
echo "    wc -l $OUTPUT_PATH/molecules/molecules.csv"
