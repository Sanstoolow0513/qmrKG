import { NextResponse } from 'next/server';
import neo4j from 'neo4j-driver';

const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USER = process.env.NEO4J_USER || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';

const DEFAULT_NODE_LIMIT = 500;
const DEFAULT_REL_LIMIT = 1500;

function parseLimit(value: string | undefined, fallback: number): number {
  if (value === undefined || value === '') {
    return fallback;
  }
  const n = parseInt(value, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
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

    const nodesResult = await session.run(`MATCH (n) RETURN n LIMIT $limit`, { limit: neo4j.int(nodeLimit) });
    const linksResult = await session.run(`MATCH ()-[r]->() RETURN r LIMIT $limit`, { limit: neo4j.int(relLimit) });

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
    const truncated = nodes.length >= nodeLimit || rawLinks.length >= relLimit;

    return NextResponse.json({
      nodes,
      links,
      truncated,
      limits: { nodes: nodeLimit, rels: relLimit },
      ...(includeTotals && totalNodes !== undefined && totalRels !== undefined
        ? { totals: { nodes: totalNodes, rels: totalRels } }
        : {}),
      meta: {
        nodesReturned: nodes.length,
        linksReturned: links.length,
        linksRaw: rawLinks.length,
        linksDroppedMissingEndpoint: droppedLinks,
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
