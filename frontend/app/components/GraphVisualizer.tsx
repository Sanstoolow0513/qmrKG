'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import type { GraphCanvasPayload } from './GraphCanvas';

const GraphCanvas = dynamic(() => import('./GraphCanvas'), {
  ssr: false,
  loading: () => <div className="p-4 text-center text-gray-500">加载图谱引擎中...</div>,
});

export default function GraphVisualizer() {
  const [payload, setPayload] = useState<GraphCanvasPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await fetch('/api/graph');
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.details || data.error || 'Failed to fetch graph data');
        }

        setPayload(data);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Error loading graph');
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
  }, []);

  if (loading) {
    return <div className="flex h-full items-center justify-center p-8">连接 Neo4j 并获取数据中...</div>;
  }
  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-red-500 bg-red-50 dark:bg-red-950/20 rounded-md">
        错误: {error}
      </div>
    );
  }
  if (!payload || payload.nodes.length === 0) {
    return <div className="flex h-full items-center justify-center p-8">数据库中暂无数据或查询结果为空。</div>;
  }

  return <GraphCanvas payload={payload} />;
}
