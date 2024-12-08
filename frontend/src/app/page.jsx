import TextGenerator from "../components/TextGenerator.tsx";

export default function Home() {
  return (
    <main className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-8">
        LLaMA Text Generator
      </h1>
      <TextGenerator />
    </main>
  );
}