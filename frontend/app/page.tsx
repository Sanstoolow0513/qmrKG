import GraphVisualizer from "./components/GraphVisualizer";

export default function Home() {
  return (
    <div className="relative w-screen h-screen overflow-hidden bg-white dark:bg-[#111] font-sans">
      <div className="absolute top-4 left-6 z-20 pointer-events-none">
        <h1 className="text-2xl font-bold text-black dark:text-zinc-50 mb-1 drop-shadow-sm">
          QmrKG 知识图谱展示
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400 drop-shadow-sm">
          基于 Next.js 与 Neo4j 的可视化实现
        </p>
      </div>
      <div className="absolute inset-0 z-10">
        <GraphVisualizer />
      </div>
    </div>
  );
}
