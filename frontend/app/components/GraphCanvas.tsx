'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { forceCollide } from 'd3-force';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import {
  colorForNodeLabel,
  colorForRelationType,
  ENTITY_NODE_COLORS,
  RELATION_COLORS,
} from '../lib/graphTheme';

// 普通节点显示标签的最小缩放倍数（放大到此比例才显示）
const LABEL_ZOOM_MIN = 1.4;
// 节点大小范围：最小/最大半径（对数映射，略小以减轻拥挤）
const NODE_R_MIN = 2.5;
const NODE_R_MAX = 11;
// 始终显示标签的节点半径阈值（与旧 15.5/18 比例对齐，仅最高频少数节点）
const ALWAYS_SHOW_LABEL_R = 9.5;
// 多边曲率分配序列
const CURVATURE_OFFSETS = [0, 0.3, -0.3, 0.5, -0.5];

interface NodeData {
  id: number;
  label: string;
  name?: string;
  title?: string;
  description?: string;
  frequency?: number;
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
  frequency?: number;
  evidences?: string;
  _curvature?: number;
  [key: string]: unknown;
}

export interface GraphCanvasPayload {
  nodes: NodeData[];
  links: LinkData[];
  truncated?: boolean;
  /** 未抽样或未设置上限时可为 undefined；字段可为 null 表示该维度无上限 */
  limits?: { nodes: number | null; rels: number | null };
  totals?: { nodes: number; rels: number };
  meta?: {
    nodesReturned: number;
    linksReturned: number;
    linksRaw: number;
    linksDroppedMissingEndpoint: number;
    maxNodeFrequency?: number;
    maxLinkFrequency?: number;
  };
}

function parseEvidences(raw: string | undefined): string[] {
  if (!raw) return [];
  try { return JSON.parse(raw) as string[]; } catch { return [raw]; }
}

function getNodeName(node: unknown): string {
  if (!node) return '';
  if (typeof node === 'number') return String(node);
  const n = node as NodeData;
  return String(n.name ?? n.title ?? n.label ?? n.id);
}

export default function GraphCanvas({ payload }: { payload: GraphCanvasPayload }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<NodeData, LinkData> | undefined>(undefined);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const [hiddenNodes, setHiddenNodes] = useState<Set<string>>(new Set());
  const [hiddenRels, setHiddenRels] = useState<Set<string>>(new Set());
  const [hoveredNode, setHoveredNode] = useState<NodeData | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);
  const [selectedLink, setSelectedLink] = useState<LinkData | null>(null);

  const maxNodeFreq = Math.max(payload.meta?.maxNodeFrequency ?? 1, 1);
  const maxLinkFreq = Math.max(payload.meta?.maxLinkFrequency ?? 1, 1);

  const getNodeRadius = useCallback((node: NodeData): number => {
    const freq = typeof node.frequency === 'number' ? node.frequency : 1;
    return NODE_R_MIN + (NODE_R_MAX - NODE_R_MIN) * (Math.log(freq + 1) / Math.log(maxNodeFreq + 1));
  }, [maxNodeFreq]);

  /** 与 hover 放大后圆一致，并留边距，供碰撞与点击热区共用 */
  const getCollisionRadius = useCallback(
    (node: NodeData) => getNodeRadius(node) * 1.2 + 2,
    [getNodeRadius],
  );

  const graphData = useMemo(() => {
    const validNodeIds = new Set<number>();
    const nodes = payload.nodes.filter(node => {
      if (hiddenNodes.has(node.label)) return false;
      validNodeIds.add(node.id);
      return true;
    });

    // 先过滤，再按节点对分配曲率
    const filtered = payload.links.filter(link => {
      if (hiddenRels.has(link.type)) return false;
      const srcId = typeof link.source === 'object' ? (link.source as NodeData).id : link.source;
      const tgtId = typeof link.target === 'object' ? (link.target as NodeData).id : link.target;
      return validNodeIds.has(srcId as number) && validNodeIds.has(tgtId as number);
    });

    // 统计每个无向节点对的边数，分配曲率偏移
    const pairCount = new Map<string, number>();
    const links = filtered.map(link => {
      const srcId = typeof link.source === 'object' ? (link.source as NodeData).id : link.source;
      const tgtId = typeof link.target === 'object' ? (link.target as NodeData).id : link.target;
      const a = Math.min(srcId as number, tgtId as number);
      const b = Math.max(srcId as number, tgtId as number);
      const key = `${a}-${b}`;
      const idx = pairCount.get(key) ?? 0;
      pairCount.set(key, idx + 1);
      return { ...link, _curvature: CURVATURE_OFFSETS[idx] ?? 0.6 };
    });

    return { nodes, links };
  }, [payload.nodes, payload.links, hiddenNodes, hiddenRels]);

  // 构建邻接表，用于 hover 暗化（以节点数字 ID 为 key）
  const adjacencyMap = useMemo(() => {
    const map = new Map<number, Set<number>>();
    for (const link of graphData.links) {
      const srcId = typeof link.source === 'object' ? (link.source as NodeData).id : link.source as number;
      const tgtId = typeof link.target === 'object' ? (link.target as NodeData).id : link.target as number;
      if (!map.has(srcId)) map.set(srcId, new Set());
      if (!map.has(tgtId)) map.set(tgtId, new Set());
      map.get(srcId)!.add(tgtId);
      map.get(tgtId)!.add(srcId);
    }
    return map;
  }, [graphData.links]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setDimensions({ width: el.offsetWidth, height: el.offsetHeight || 600 });
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => { ro.disconnect(); window.removeEventListener('resize', measure); };
  }, []);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    const chargeForce = fg.d3Force('charge');
    if (chargeForce) {
      const cf = chargeForce as { strength?: (n: number) => unknown; distanceMax?: (n: number) => unknown };
      if (typeof cf.strength === 'function') {
        cf.strength(-55);
      }
      if (typeof cf.distanceMax === 'function') {
        cf.distanceMax(Math.max(dimensions.width, dimensions.height) * 2);
      }
    }

    fg.d3Force(
      'collide',
      forceCollide<NodeData>()
        .radius((d) => getCollisionRadius(d))
        .iterations(2),
    );
  }, [graphData, dimensions.width, dimensions.height, getCollisionRadius]);

  const onEngineStop = useCallback(() => { fgRef.current?.pauseAnimation(); }, []);
  const resumeFrame = useCallback(() => { fgRef.current?.resumeAnimation(); }, []);

  const visibleNodeCount = graphData.nodes.length;
  const visibleLinkCount = graphData.links.length;
  const loadedNodes = payload.meta?.nodesReturned;
  const loadedLinks = payload.meta?.linksReturned;
  const filterActive =
    loadedNodes != null &&
    loadedLinks != null &&
    (loadedNodes !== visibleNodeCount || loadedLinks !== visibleLinkCount);

  return (
    <div className="relative w-full h-full">
      {/* 当前画布可见节点/关系数（随类型过滤变化） */}
      <div className="absolute top-24 left-6 z-20 max-w-md pointer-events-none">
        <p className="text-xs text-zinc-700 dark:text-zinc-300 bg-white/95 dark:bg-zinc-900/95 border border-zinc-200 dark:border-zinc-700 rounded-md px-3 py-2 shadow-sm backdrop-blur-sm">
          <span className="text-zinc-500 dark:text-zinc-500">当前展示：</span>
          <span className="font-medium tabular-nums text-zinc-900 dark:text-zinc-100">{visibleNodeCount}</span>
          <span className="text-zinc-500 dark:text-zinc-500"> 个节点 · </span>
          <span className="font-medium tabular-nums text-zinc-900 dark:text-zinc-100">{visibleLinkCount}</span>
          <span className="text-zinc-500 dark:text-zinc-500"> 条关系</span>
          {filterActive && loadedNodes != null && loadedLinks != null && (
            <span className="block mt-1 text-zinc-500 dark:text-zinc-500">
              已加载 {loadedNodes} 个节点、{loadedLinks} 条关系（部分类型已隐藏）
            </span>
          )}
        </p>
      </div>

      {/* 截断提示 */}
      {(payload.truncated || (payload.meta != null && payload.meta.linksDroppedMissingEndpoint > 0)) && (
        <div className="absolute bottom-6 left-6 z-20 max-w-xl pointer-events-none">
          <p className="text-xs text-amber-800 dark:text-amber-200/90 bg-amber-50/95 dark:bg-amber-950/80 border border-amber-200/80 dark:border-amber-800/60 rounded-md px-3 py-2 shadow-sm backdrop-blur-sm pointer-events-auto">
            {payload.truncated && (
              <>
                当前为抽样子图（节点上限 {payload.limits?.nodes ?? '未限制'}，关系上限 {payload.limits?.rels ?? '未限制'}）。
              </>
            )}
            {payload.totals != null && <>{' '}数据库总计约 {payload.totals.nodes} 个节点、{payload.totals.rels} 条关系。</>}
            {payload.meta != null && payload.meta.linksDroppedMissingEndpoint > 0 && (
              <>{' '}已丢弃 {payload.meta.linksDroppedMissingEndpoint} 条端点不在本批节点内的关系。</>
            )}
          </p>
        </div>
      )}

      {/* 节点详情面板 */}
      {selectedNode && (
        <div className="absolute left-3 bottom-3 z-20 w-72 max-h-80 overflow-y-auto rounded-md border border-zinc-200 dark:border-zinc-700 bg-white/95 dark:bg-zinc-900/95 px-3 py-2.5 text-xs text-zinc-700 dark:text-zinc-300 shadow-md">
          <div className="flex items-start justify-between gap-2 mb-2">
            <span className="font-bold text-sm text-zinc-900 dark:text-zinc-100 leading-tight break-all">
              {getNodeName(selectedNode)}
            </span>
            <span
              className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium text-white"
              style={{ backgroundColor: colorForNodeLabel(selectedNode.label) }}
            >
              {selectedNode.label}
            </span>
          </div>
          {selectedNode.description && (
            <p className="text-zinc-600 dark:text-zinc-400 mb-2 leading-relaxed">
              {String(selectedNode.description)}
            </p>
          )}
          {selectedNode.frequency != null && (
            <div className="mb-2">
              <span className="text-zinc-500 dark:text-zinc-500">频次：</span>
              <span className="font-medium">{String(selectedNode.frequency)}</span>
              <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-100 dark:bg-zinc-800 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    backgroundColor: colorForNodeLabel(selectedNode.label),
                    width: `${Math.min(100, (Math.log((selectedNode.frequency as number) + 1) / Math.log(maxNodeFreq + 1)) * 100)}%`,
                  }}
                />
              </div>
            </div>
          )}
          <button
            onClick={() => setSelectedNode(null)}
            className="mt-1 text-[10px] text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
          >
            关闭
          </button>
        </div>
      )}

      {/* 关系详情面板 */}
      {selectedLink && !selectedNode && (
        <div className="absolute left-3 bottom-3 z-20 w-72 max-h-96 overflow-y-auto rounded-md border border-zinc-200 dark:border-zinc-700 bg-white/95 dark:bg-zinc-900/95 px-3 py-2.5 text-xs text-zinc-700 dark:text-zinc-300 shadow-md">
          <div className="flex items-center gap-2 mb-2">
            <span
              className="rounded px-1.5 py-0.5 text-[10px] font-medium text-white"
              style={{ backgroundColor: colorForRelationType(selectedLink.type) }}
            >
              {selectedLink.type}
            </span>
            {selectedLink.frequency != null && (
              <span className="text-zinc-500">频次 {String(selectedLink.frequency)}</span>
            )}
          </div>
          <div className="mb-2 text-zinc-600 dark:text-zinc-400">
            <span className="font-medium text-zinc-800 dark:text-zinc-200">
              {getNodeName(selectedLink.source)}
            </span>
            <span className="mx-1">→</span>
            <span className="font-medium text-zinc-800 dark:text-zinc-200">
              {getNodeName(selectedLink.target)}
            </span>
          </div>
          {parseEvidences(selectedLink.evidences).length > 0 && (
            <div>
              <div className="font-semibold text-zinc-700 dark:text-zinc-300 mb-1">原文依据</div>
              <ul className="space-y-1.5">
                {parseEvidences(selectedLink.evidences).map((ev, i) => (
                  <li
                    key={i}
                    className="border-l-2 pl-2 text-zinc-500 dark:text-zinc-400 leading-relaxed"
                    style={{ borderColor: colorForRelationType(selectedLink.type) }}
                  >
                    {ev.length > 150 ? ev.slice(0, 150) + '…' : ev}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <button
            onClick={() => setSelectedLink(null)}
            className="mt-2 text-[10px] text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
          >
            关闭
          </button>
        </div>
      )}

      <div
        ref={containerRef}
        onPointerMove={resumeFrame}
        className="absolute inset-0 bg-transparent"
      >
        {/* 过滤面板 */}
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
                        if (e.target.checked) next.delete(name); else next.add(name);
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
                        if (e.target.checked) next.delete(name); else next.add(name);
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
          /* 与 getNodeRadius 一致：库内半径 = sqrt(nodeVal)*nodeRelSize，用于边/箭头端点与圆周对齐 */
          nodeRelSize={1}
          nodeVal={(node: NodeData) => getNodeRadius(node) ** 2}
          autoPauseRedraw
          d3AlphaDecay={0.052}
          d3VelocityDecay={0.68}
          warmupTicks={0}
          cooldownTicks={100}
          onEngineStop={onEngineStop}
          onZoom={resumeFrame}
          onNodeDrag={resumeFrame}
          onNodeDragEnd={resumeFrame}
          onNodeHover={(node) => { setHoveredNode(node ?? null); resumeFrame(); }}
          onNodeClick={(node) => { setSelectedNode(node); setSelectedLink(null); resumeFrame(); }}
          onLinkClick={(link) => { setSelectedLink(link as unknown as LinkData); setSelectedNode(null); resumeFrame(); }}
          onBackgroundClick={() => { setSelectedNode(null); setSelectedLink(null); }}
          nodeLabel={(node: NodeData) => `${node.label}: ${node.name || node.title || node.id}`}
          nodePointerAreaPaint={(node: NodeData, color, ctx) => {
            const r = getCollisionRadius(node);
            ctx.beginPath();
            ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
          }}
          nodeCanvasObject={(node: NodeData, ctx, globalScale) => {
            const r = getNodeRadius(node);
            const isHovered = hoveredNode?.id === node.id;
            const hasHover = hoveredNode !== null;
            const isNeighbor = hasHover && (adjacencyMap.get(hoveredNode!.id)?.has(node.id) ?? false);
            const dimmed = hasHover && !isHovered && !isNeighbor;

            ctx.globalAlpha = dimmed ? 0.15 : 1;

            // 节点圆形
            const drawR = isHovered ? r * 1.2 : r;
            ctx.beginPath();
            ctx.arc(node.x!, node.y!, drawR, 0, 2 * Math.PI, false);
            ctx.fillStyle = colorForNodeLabel(node.label);
            ctx.fill();

            // Hover 时白色外圈
            if (isHovered) {
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, drawR + 1.5, 0, 2 * Math.PI, false);
              ctx.strokeStyle = 'rgba(255,255,255,0.9)';
              ctx.lineWidth = 1.5 / globalScale;
              ctx.stroke();
            }

            // 标签：极高频节点始终显示，其他节点仅在放大时显示
            const alwaysShow = r > ALWAYS_SHOW_LABEL_R;
            if (!alwaysShow && globalScale < LABEL_ZOOM_MIN) {
              ctx.globalAlpha = 1;
              return;
            }

            const label = String(node.name ?? node.title ?? node.label ?? node.id);
            const fontSize = Math.max(8 / globalScale, 2.5);
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const tw = ctx.measureText(label).width;
            const pad = 1.5 / globalScale;
            const labelY = node.y! + drawR + fontSize * 0.65 + pad;

            // 深色半透明背景（深色模式友好）
            ctx.fillStyle = dimmed ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0.55)';
            ctx.fillRect(
              node.x! - tw / 2 - pad,
              labelY - fontSize * 0.6 - pad,
              tw + pad * 2,
              fontSize + pad * 2,
            );

            // 标签文字
            ctx.fillStyle = dimmed ? 'rgba(200,200,200,0.5)' : '#e8e8e8';
            ctx.fillText(label, node.x!, labelY);

            ctx.globalAlpha = 1;
          }}
          linkColor={(link: LinkData) => colorForRelationType(link.type)}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkDirectionalArrowColor={(link: LinkData) => colorForRelationType(link.type)}
          linkDirectionalParticles={0}
          linkWidth={(link: LinkData) => 0.6 + ((link.frequency ?? 1) / maxLinkFreq) * 3.4}
          linkCurvature={(link: LinkData) => link._curvature ?? 0}
          linkLabel="type"
        />
      </div>
    </div>
  );
}
