// 14-next-steps/tying-it-together.ts

/**
 * Module 14: Tying it Together & Next Steps
 * A conceptual example showing integration of various TypeScript features.
 */

console.log("--- Module 14: Tying it Together ---");

// Assume these types/interfaces might be imported from './types.js'
// Using features: interface, type alias, union, readonly, optional
interface DocumentChunk {
  readonly id: string;
  documentId: string;
  text: string;
  score: number;
}
type RagStatus = "Idle" | "Processing" | "Success" | "Error";
type RagConfig = Readonly<{ // Using Utility Type
  model: string;
  topK: number;
}>;

// Assume this class might be imported from './processor.js'
// Using features: class, constructor, private/public, async method, generics, error handling
class SimpleRAGProcessor<T extends DocumentChunk> {
  // Using parameter property shorthand for config
  constructor(
    private readonly config: RagConfig,
    private status: RagStatus = "Idle"
  ) {
    console.log(`[RAG Processor] Initialized. Model: ${this.config.model}, Status: ${this.status}`);
  }

  // Public async method using generics and error handling
  public async generate(query: string, retrieveDocsFn: () => Promise<T[]>): Promise<string | null> {
    if (this.status === "Processing") {
      throw new Error("Processor is already busy.");
    }
    this.updateStatus("Processing");
    console.log(`[RAG Processor] Processing query: "${query}"`);

    try {
      // Simulate retrieval (using imported async function)
      console.log("[RAG Processor] Retrieving documents...");
      const docs = await retrieveDocsFn(); // Uses await
      console.log(`[RAG Processor] Retrieved ${docs.length} documents.`);

      // Filter top K (using class config)
      const topDocs = docs.sort((a, b) => b.score - a.score).slice(0, this.config.topK);

      // Simulate generation (using imported utility function)
      const context = SimpleRagUtils.formatContext(topDocs); // Using static method
      console.log(`[RAG Processor] Generating response with ${topDocs.length} docs...`);
      await SimpleRagUtils.simulateDelay(500); // Simulate async work

      const response = `Response for "${query}" based on context: ${context.substring(0, 50)}...`;
      this.updateStatus("Success");
      return response;

    } catch (error) {
      this.updateStatus("Error");
      console.error("[RAG Processor] Error during generation:", SimpleRagUtils.getErrorMessage(error)); // Using utility
      return null; // Return null on failure
    } finally {
      // Finally block for cleanup or logging regardless of success/error
      console.log(`[RAG Processor] Processing finished with status: ${this.status}`);
      if (this.status !== "Error") this.updateStatus("Idle"); // Reset if not errored
    }
  }

  // Private method
  private updateStatus(newStatus: RagStatus): void {
    this.status = newStatus;
    // console.log(`[RAG Processor] Status updated to: ${this.status}`); // Optional logging
  }
}

// Assume these might be imported from './utils.js'
// Using features: static class members, function, type guard
class SimpleRagUtils {
  static formatContext(docs: DocumentChunk[]): string {
    if (docs.length === 0) return "No context.";
    return docs.map(d => `[${d.id}: ${d.score.toFixed(2)}] ${d.text}`).join("\n");
  }

  static async simulateDelay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  static getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }
}

// --- Main Execution Logic ---
// Using features: async IIFE, instantiation, function call, .then/.catch

// Define a simple async function to simulate document retrieval
async function fakeDocRetrieval(): Promise<DocumentChunk[]> {
  await SimpleRagUtils.simulateDelay(300);
  // Simulate finding docs
  return [
    { id: "chunk-001", documentId: "doc-A", text: "TypeScript is great for large projects.", score: 0.9 },
    { id: "chunk-002", documentId: "doc-B", text: "Async/await simplifies promises.", score: 0.85 },
    { id: "chunk-003", documentId: "doc-A", text: "Generics provide type safety.", score: 0.92 },
  ];
}

// Immediately Invoked Function Expression (IIFE) to allow top-level await
(async () => {
  console.log("--- Running Integrated Example ---");

  const config: RagConfig = { model: "MiniRAG-TS", topK: 2 };
  const processor = new SimpleRAGProcessor(config);

  // Call the async method and handle the result promise
  processor.generate("Tell me about TypeScript", fakeDocRetrieval)
    .then(response => {
      if (response) {
        console.log("\n--- FINAL RESPONSE ---");
        console.log(response);
      } else {
        console.log("\n--- PROCESSING FAILED ---");
      }
    })
    .catch(outerError => {
      // Catch errors thrown *synchronously* by the generate call itself (e.g., if busy)
      console.error("\n--- SYNCHRONOUS ERROR ---");
      console.error(SimpleRagUtils.getErrorMessage(outerError));
    });

})();

// --- End of Module 14 ---