/** 与 src/qmrkg/kg_schema.py 中 ENTITY_TYPE_LABELS / RELATION_TYPE_LABELS 对齐 */

export const ENTITY_NODE_COLORS: Record<string, string> = {
  Protocol: '#2563eb',
  Concept: '#16a34a',
  Mechanism: '#ca8a04',
  Metric: '#9333ea',
};

export const RELATION_COLORS: Record<string, string> = {
  CONTAINS: '#64748b',
  DEPENDS_ON: '#0ea5e9',
  COMPARED_WITH: '#f97316',
  APPLIED_TO: '#ec4899',
};

const DEFAULT_NODE = '#6b7280';
const DEFAULT_LINK = '#94a3b8';

function hashHue(input: string): number {
  let h = 0;
  for (let i = 0; i < input.length; i++) {
    h = input.charCodeAt(i) + ((h << 5) - h);
  }
  return Math.abs(h) % 360;
}

export function colorForNodeLabel(label: string): string {
  if (ENTITY_NODE_COLORS[label]) {
    return ENTITY_NODE_COLORS[label];
  }
  const hue = hashHue(label);
  return `hsl(${hue} 42% 46%)`;
}

export function colorForRelationType(type: string): string {
  if (RELATION_COLORS[type]) {
    return RELATION_COLORS[type];
  }
  return DEFAULT_LINK;
}

export const DEFAULT_NODE_COLOR = DEFAULT_NODE;
