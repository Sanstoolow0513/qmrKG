#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# raw 文件太多 (12000+)，打包上传
echo "=== 打包 triples/raw → triples-raw.tar.gz ==="
tar -czf /tmp/qmrkg-triples-raw.tar.gz -C data/triples raw
echo "打包完成: $(du -h /tmp/qmrkg-triples-raw.tar.gz | cut -f1)"

echo "=== 上传 triples-raw.tar.gz ==="
rclone copy /tmp/qmrkg-triples-raw.tar.gz GoogleDrive:qmrkg-data/triples-raw.tar.gz --progress
rm /tmp/qmrkg-triples-raw.tar.gz

# merged 只有 4 个文件，直接同步
echo ""
echo "=== 上传 triples/merged ==="
rclone sync data/triples/merged GoogleDrive:qmrkg-data/triples-merged --progress

echo ""
echo "完成。"
