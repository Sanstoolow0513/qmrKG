#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 下载 triples-raw.tar.gz ==="
rclone copy GoogleDrive:qmrkg-data/triples-raw.tar.gz /tmp/qmrkg-triples-raw.tar.gz --progress

echo "=== 解压到 data/triples/raw ==="
rm -rf data/triples/raw
tar -xzf /tmp/qmrkg-triples-raw.tar.gz -C data/triples
rm /tmp/qmrkg-triples-raw.tar.gz

echo ""
echo "=== 下载 triples/merged ==="
rclone sync GoogleDrive:qmrkg-data/triples-merged data/triples/merged --progress

echo ""
echo "完成。"
