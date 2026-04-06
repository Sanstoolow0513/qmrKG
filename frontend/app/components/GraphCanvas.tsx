'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import {
  colorForNodeLabel,
  colorForRelationType,
  ENTITY_NODE_COLORS,
  RELATION_COLORS,
} from '../lib/graphTheme';

const LABEL_ZOOM_MIN = 0.85;
const NODE_RADIUS = 5;

interface NodeData {
  id: number;
  label: string;
  name?: string;
  title?: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  [key: string]: unknown;
}

interface LinkData {
  source: number;
  target: number;
  type: string;
}

export interface GraphCanvasPayload {
  nodes: NodeData[];
  links: LinkData[];
  truncated?: boolean;
  limits?: { nodes: number; rels: number };
  totals?: { nodes: number; rels: number };
  meta?: {
    nodesReturned: number;
    linksReturned: number;
    linksRaw: number;
    linksDroppedMissingEndpoint: number;
  };
}

export default function GraphCanvas({ payload }: { payload: GraphCanvasPayload }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<NodeData, LinkData> | undefined>(undefined);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const [hiddenNodes, setHiddenNodes] = useState<Set<string>>(new Set());
  const [hiddenRels, setHiddenRels] = useState<Set<string>>(new Set());

  const graphData = useMemo(() => {
    const validNodeIds = new Set();
    const nodes = payload.nodes.filter(node => {
      if (hiddenNodes.has(node.label)) return false;
      validNodeIds.add(node.id);
      return true;
    });
    
    const links = payload.links.filter(link => {
      if (hiddenRels.has(link.type)) return false;
      
      const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source;
      const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target;
      
      return validNodeIds.has(sourceId) && validNodeIds.has(targetId);
    });
    
    return { nodes, links };
  }, [payload.nodes, payload.links, hiddenNodes, hiddenRels]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) {
      return;
    }

    const measure = () => {
      setDimensions({
        width: el.offsetWidth,
        height: el.offsetHeight || 600,
      });
    };

    measure();

    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  useEffect(() => {
    const fg = fgRef.current;
    if (fg) {
      // 限制排斥力（charge）的最大作用距离。
      // 避免没有连边的孤立节点被中心图谱的巨大排斥力推挤到无限远，从而在边缘形成一个大圆圈。
      const chargeForce = fg.d3Force('charge');
      if (chargeForce && typeof (chargeForce as any).distanceMax === 'function') {
        (chargeForce as any).distanceMax(150);
      }
    }
  }, [graphData]);

  const onEngineStop = useCallback(() => {
    fgRef.current?.pauseAnimation();
  }, []);

  const resumeFrame = useCallback(() => {
    fgRef.current?.resumeAnimation();
  }, []);

  return (
    <div className="relative w-full h-full">
      {(payload.truncated ||
        (payload.meta != null && payload.meta.linksDroppedMissingEndpoint > 0)) && (
        <div className="absolute bottom-6 left-6 z-20 max-w-xl pointer-events-none">
          <p className="text-xs text-amber-800 dark:text-amber-200/90 bg-amber-50/95 dark:bg-amber-950/80 border border-amber-200/80 dark:border-amber-800/60 rounded-md px-3 py-2 shadow-sm backdrop-blur-sm pointer-events-auto">
            {payload.truncated && (
              <>
                当前为抽样子图（节点上限 {payload.limits?.nodes ?? '—'}，关系上限 {payload.limits?.rels ?? '—'}）。
              </>
            )}
            {payload.totals != null && (
              <>
                {' '}
                数据库总计约 {payload.totals.nodes} 个节点、{payload.totals.rels} 条关系。
              </>
            )}
            {payload.meta != null && payload.meta.linksDroppedMissingEndpoint > 0 && (
              <>
                {' '}
                已丢弃 {payload.meta.linksDroppedMissingEndpoint} 条端点不在本批节点内的关系。
              </>
            )}
          </p>
        </div>
      )}
      <div
        ref={containerRef}
        onPointerMove={resumeFrame}
        className="absolute inset-0 bg-transparent"
      >
        <div className="absolute right-2 top-2 z-10 max-w-[min(100%,280px)] rounded-md border border-zinc-200 dark:border-zinc-700 bg-white/95 dark:bg-zinc-900/95 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300 shadow-sm">
          <div className="font-semibold mb-1.5 text-zinc-900 dark:text-zinc-100 flex justify-between items-center">
            <span>节点类型过滤</span>
            {(hiddenNodes.size > 0 || hiddenRels.size > 0) && (
              <button
                onClick={() => { setHiddenNodes(new Set()); setHiddenRels(new Set()); }}
                className="text-[10px] text-blue-500 hover:text-blue-600 dark:text-blue-400"
              >
                重置
              </button>
            )}
          </div>
          <ul className="space-y-1 mb-2">
            {Object.entries(ENTITY_NODE_COLORS).map(([name, hex]) => (
              <li key={name} className="flex items-center gap-2">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="accent-blue-600 w-3 h-3 cursor-pointer"
                    checked={!hiddenNodes.has(name)}
                    onChange={(e) => {
                      setHiddenNodes(prev => {
                        const next = new Set(prev);
                        if (e.target.checked) next.delete(name);
                        else next.add(name);
                        return next;
                      });
                    }}
                  />
                  <span
                    className="inline-block h-3 w-3 shrink-0 rounded-full border border-zinc-300/80 dark:border-zinc-600"
                    style={{ backgroundColor: hex, opacity: hiddenNodes.has(name) ? 0.3 : 1 }}
                  />
                  <span className={hiddenNodes.has(name) ? 'opacity-50 line-through' : ''}>{name}</span>
                </label>
              </li>
            ))}
          </ul>
          <div className="font-semibold mb-1.5 text-zinc-900 dark:text-zinc-100">关系类型过滤</div>
          <ul className="space-y-1">
            {Object.entries(RELATION_COLORS).map(([name, hex]) => (
              <li key={name} className="flex items-center gap-2">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="accent-blue-600 w-3 h-3 cursor-pointer"
                    checked={!hiddenRels.has(name)}
                    onChange={(e) => {
                      setHiddenRels(prev => {
                        const next = new Set(prev);
                        if (e.target.checked) next.delete(name);
                        else next.add(name);
                        return next;
                      });
                    }}
                  />
                  <span
                    className="inline-block h-1.5 w-3 shrink-0 rounded-sm border border-zinc-300/80 dark:border-zinc-600"
                    style={{ backgroundColor: hex, opacity: hiddenRels.has(name) ? 0.3 : 1 }}
                  />
                  <span className={`break-all ${hiddenRels.has(name) ? 'opacity-50 line-through' : ''}`}>{name}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>

        <ForceGraph2D<NodeData, LinkData>
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          autoPauseRedraw
          d3AlphaDecay={0.052}
          d3VelocityDecay={0.68}
          warmupTicks={0}
          cooldownTicks={80}
          onEngineStop={onEngineStop}
          onZoom={resumeFrame}
          onNodeDrag={resumeFrame}
          onNodeDragEnd={resumeFrame}
          nodeLabel={(node: NodeData) => `${node.label}: ${node.name || node.title || node.id}`}
          nodePointerAreaPaint={(node: NodeData, color, ctx) => {
            ctx.beginPath();
            ctx.arc(node.x!, node.y!, NODE_RADIUS, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
          }}
          nodeCanvasObject={(node: NodeData, ctx, globalScale) => {
            const fill = colorForNodeLabel(node.label);
            ctx.beginPath();
            ctx.arc(node.x!, node.y!, NODE_RADIUS, 0, 2 * Math.PI, false);
            ctx.fillStyle = fill;
            ctx.fill();

            if (globalScale < LABEL_ZOOM_MIN) {
              return;
            }
            const label = String(node.name ?? node.title ?? node.label ?? node.id);
            const fontSize = Math.max(10 / globalScale, 3);
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#888';
            ctx.fillText(label, node.x!, node.y! + NODE_RADIUS + fontSize * 0.65);
          }}
          linkColor={(link: LinkData) => colorForRelationType(link.type)}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkDirectionalArrowColor={(link: LinkData) => colorForRelationType(link.type)}
          linkDirectionalParticles={0}
          linkWidth={0.8}
          linkLabel="type"
        />
      </div>
    </div>
  );
}
