import { NextResponse } from 'next/server';
import neo4j from 'neo4j-driver';

const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USER = process.env.NEO4J_USER || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';

/**
 * 未设置 env 时的默认抽样规模（连贯子图：先 Top-K 节点，再只取这些节点之间的边）。
 * 全量加载：设置 NEO4J_GRAPH_NODE_LIMIT=0 且 NEO4J_GRAPH_REL_LIMIT=0。
 */
const DEFAULT_NODE_LIMIT = 1000;
const DEFAULT_REL_LIMIT = 4000;

/**
 * 解析节点/关系条数上限。正整数为上限；0 或负数表示不限制。
 * 未设置环境变量时使用 defaultLimit。
 */
function parseLimit(value: string | undefined, defaultLimit: number | null): number | null {
  if (value === undefined || value === '') {
    return defaultLimit;
  }
  const n = parseInt(value, 10);
  if (!Number.isFinite(n)) {
    return defaultLimit;
  }
  if (n <= 0) {
    return null;
  }
  return n;
}

const driver = neo4j.driver(
  NEO4J_URI,
  neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD)
);

export async function GET() {
  const nodeLimit = parseLimit(process.env.NEO4J_GRAPH_NODE_LIMIT, DEFAULT_NODE_LIMIT);
  const relLimit = parseLimit(process.env.NEO4J_GRAPH_REL_LIMIT, DEFAULT_REL_LIMIT);
  const includeTotals = process.env.NEO4J_GRAPH_INCLUDE_TOTALS === '1' || process.env.NEO4J_GRAPH_INCLUDE_TOTALS === 'true';

  const session = driver.session();
  try {
    let totalNodes: number | undefined;
    let totalRels: number | undefined;
    if (includeTotals) {
      const [cn, cr] = await Promise.all([
        session.run(`MATCH (n) RETURN count(n) AS c`),
        session.run(`MATCH ()-[r]->() RETURN count(r) AS c`),
      ]);
      totalNodes = cn.records[0]?.get('c')?.toNumber?.() ?? Number(cn.records[0]?.get('c'));
      totalRels = cr.records[0]?.get('c')?.toNumber?.() ?? Number(cr.records[0]?.get('c'));
    }

    const nodesQuery =
      nodeLimit === null
        ? `MATCH (n) RETURN n ORDER BY n.frequency DESC`
        : `MATCH (n) RETURN n ORDER BY n.frequency DESC LIMIT $nodeLimit`;

    const nodesResult = await session.run(
      nodesQuery,
      nodeLimit === null ? {} : { nodeLimit: neo4j.int(nodeLimit) },
    );

    const linksResult = await (async () => {
      if (nodeLimit === null) {
        const linksQuery =
          relLimit === null
            ? `MATCH ()-[r]->() RETURN r ORDER BY r.frequency DESC`
            : `MATCH ()-[r]->() RETURN r ORDER BY r.frequency DESC LIMIT $relLimit`;
        return session.run(
          linksQuery,
          relLimit === null ? {} : { relLimit: neo4j.int(relLimit) },
        );
      }
      const inducedBase = `
        MATCH (n)
        WITH n ORDER BY n.frequency DESC LIMIT $nodeLimit
        WITH collect(n) AS nodeList
        MATCH (a)-[r]->(b)
        WHERE a IN nodeList AND b IN nodeList
        RETURN r ORDER BY r.frequency DESC`;
      const inducedQuery =
        relLimit === null ? inducedBase : `${inducedBase}\n        LIMIT $relLimit`;
      return session.run(
        inducedQuery,
        relLimit === null
          ? { nodeLimit: neo4j.int(nodeLimit) }
          : { nodeLimit: neo4j.int(nodeLimit), relLimit: neo4j.int(relLimit) },
      );
    })();

    const nodesMap = new Map<number, { id: number; label: string; [key: string]: unknown }>();
    const rawLinks: Array<{
      source: number;
      target: number;
      type: string;
      [key: string]: unknown;
    }> = [];

    nodesResult.records.forEach((record) => {
      const n = record.get('n');
      const identity = n.identity.toNumber();
      const labels = n.labels as string[];
      const primary =
        labels.find((lb) => ['Protocol', 'Concept', 'Mechanism', 'Metric'].includes(lb)) ?? labels[0] ?? 'Unknown';
      nodesMap.set(identity, {
        id: identity,
        label: primary,
        ...(serializeProps(n.properties as Record<string, unknown>) as Record<string, unknown>),
      });
    });

    linksResult.records.forEach((record) => {
      const r = record.get('r');
      rawLinks.push({
        source: r.start.toNumber(),
        target: r.end.toNumber(),
        type: r.type as string,
        ...(serializeProps(r.properties as Record<string, unknown>) as Record<string, unknown>),
      });
    });

    const nodeIds = new Set(nodesMap.keys());
    const links = rawLinks.filter((l) => nodeIds.has(l.source) && nodeIds.has(l.target));
    const droppedLinks = rawLinks.length - links.length;

    const nodes = Array.from(nodesMap.values());
    const truncated =
      (nodeLimit !== null && nodes.length >= nodeLimit) ||
      (relLimit !== null && rawLinks.length >= relLimit);

    const maxNodeFrequency = nodes.reduce((max, n) => {
      const f = typeof n.frequency === 'number' ? n.frequency : 0;
      return Math.max(max, f);
    }, 0);
    const maxLinkFrequency = links.reduce((max, l) => {
      const f = typeof l.frequency === 'number' ? l.frequency : 0;
      return Math.max(max, f);
    }, 0);

    return NextResponse.json({
      nodes,
      links,
      truncated,
      limits:
        nodeLimit === null && relLimit === null
          ? undefined
          : { nodes: nodeLimit, rels: relLimit },
      ...(includeTotals && totalNodes !== undefined && totalRels !== undefined
        ? { totals: { nodes: totalNodes, rels: totalRels } }
        : {}),
      meta: {
        nodesReturned: nodes.length,
        linksReturned: links.length,
        linksRaw: rawLinks.length,
        linksDroppedMissingEndpoint: droppedLinks,
        maxNodeFrequency,
        maxLinkFrequency,
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('Neo4j query error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch graph data', details: message },
      { status: 500 }
    );
  } finally {
    await session.close();
  }
}

function serializeProps(props: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) {
    if (v !== null && typeof v === 'object' && 'toNumber' in v && typeof (v as { toNumber: () => number }).toNumber === 'function') {
      try {
        out[k] = (v as { toNumber: () => number }).toNumber();
      } catch {
        out[k] = String(v);
      }
    } else {
      out[k] = v;
    }
  }
  return out;
}
