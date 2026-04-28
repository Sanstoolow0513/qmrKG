# QmrKG Frontend

**Location:** `frontend/`

## OVERVIEW

Next.js 16 visualization frontend for the QmrKG knowledge graph. Displays Neo4j graph data using react-force-graph-2d with interactive node/edge exploration.

## STRUCTURE

```
frontend/
├── app/
│   ├── page.tsx            # Main page: GraphVisualizer component
│   ├── layout.tsx          # Root layout with fonts/metadata
│   ├── globals.css         # Global Tailwind styles
│   ├── api/graph/route.ts  # Neo4j graph data API
│   ├── components/         # React components
│   └── lib/                # Utility functions
├── public/                 # Static assets
├── package.json            # Dependencies
├── next.config.ts          # Next.js config
├── tsconfig.json           # TypeScript config
└── eslint.config.mjs       # ESLint config
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Main visualization | `app/page.tsx` | Renders GraphVisualizer |
| Graph data API | `app/api/graph/route.ts` | Queries Neo4j, returns graph data |
| Force graph component | `app/components/GraphCanvas.tsx` | react-force-graph-2d canvas |
| Visualization wrapper | `app/components/GraphVisualizer.tsx` | Controls, legend, layout |
| Theme/colors | `app/lib/graphTheme.ts` | Graph styling constants |
| Route handler | `app/api/graph/route.ts` | Neo4j query execution |

## COMMANDS

```bash
# Development
pnpm dev        # localhost:3000

# Production
pnpm build      # Build for production
pnpm start      # Start production server

# Linting
pnpm lint       # ESLint check
```

## CONVENTIONS

### TypeScript
- **Strict mode:** Enabled in tsconfig.json
- **Path alias:** `@/*` maps to `./*`
- **Target:** ES2017
- **Module:** ESNext with bundler resolution

### Styling
- **Framework:** Tailwind CSS v4
- **Config:** postcss.config.mjs with @tailwindcss/postcss
- **Global styles:** globals.css

### Components
- Use functional components with hooks
- Props typed with interfaces
- Client components marked with `'use client'` when needed

### Data Fetching
- API routes in `app/api/` directory
- Neo4j driver for database queries
- Return JSON with nodes/edges arrays

## DEPENDENCIES

| Package | Purpose |
|---------|---------|
| next | Next.js framework |
| react/react-dom | React 19 |
| react-force-graph-2d | Graph visualization |
| d3-force | Force simulation |
| neo4j-driver | Neo4j connectivity |
| tailwindcss | Styling |

## ENVIRONMENT

Create `frontend/.env.local`:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_GRAPH_NODE_LIMIT=1000
NEO4J_GRAPH_REL_LIMIT=4000
```

## NOTES

- **Not a monorepo:** Frontend is independent project
- **Package manager:** pnpm (see pnpm-workspace.yaml)
- **Next.js version:** 16.2.2
- **React version:** 19.2.4
- **Dev indicators:** Disabled in next.config.ts
