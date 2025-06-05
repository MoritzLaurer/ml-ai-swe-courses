// 05-interfaces/interfaces.ts

/**
 * Module 5: Interfaces
 * Covers: Defining object shapes using `interface`, extending interfaces,
 * declaration merging, and comparing `interface` vs `type` for objects.
 */

console.log("--- Module 5: Interfaces ---");

// Besides `type` aliases, `interface` is another powerful way to define the
// structure or "shape" that an object should have. Interfaces are primarily
// focused on describing object shapes.



// ==========================================================================
// 1. Basic `interface` Declaration
// ==========================================================================
console.log("\n--- 1. Basic `interface` Declaration ---");

// Syntax: interface InterfaceName { propertyName: type; ... }
// Notice there's no '=' like with `type` aliases.

interface Document {
  readonly id: string; // Use readonly for immutable properties
  title: string;
  content: string;
  wordCount: number;
  author?: string;   // Use ? for optional properties
  lastModified?: Date; // Optional Date
}

// Using the interface to type an object:
const doc1: Document = {
  id: "doc-iface-001",
  title: "Interfaces in TypeScript",
  content: "Interfaces define contracts for object shapes...",
  wordCount: 450,
  // author is optional, so it's okay to omit it
  lastModified: new Date("2025-04-18")
};

console.log("Document object typed with Interface:", doc1);

// Similar to `type`, interfaces enforce the shape:
// const doc2: Document = { // Error: Property 'content' is missing...
//   id: "doc-iface-002",
//   title: "Type Aliases",
//   wordCount: 300
// };
// doc1.id = "new-id"; // Error: Cannot assign to 'id' because it is a read-only property.



// ==========================================================================
// 2. Extending Interfaces
// ==========================================================================
console.log("\n--- 2. Extending Interfaces ---");

// Interfaces can inherit properties from other interfaces using the `extends` keyword.
// This promotes reusability and building complex types from simpler ones.

interface DatedRecord {
  createdAt: Date;
  updatedAt: Date;
}

// `WebDocument` now includes all properties from `Document` AND `DatedRecord`.
interface WebDocument extends Document, DatedRecord {
  url: string;
  isIndexed: boolean;
}

const webDoc: WebDocument = {
  // Properties from Document
  id: "web-doc-001",
  title: "TypeScript Handbook - Interfaces",
  content: "The handbook explains interfaces...",
  wordCount: 1200,
  // author is optional
  // Properties from DatedRecord
  createdAt: new Date("2025-01-10"),
  updatedAt: new Date("2025-04-15"),
  // Properties specific to WebDocument
  url: "https://www.typescriptlang.org/docs/handbook/2/objects.html",
  isIndexed: true
};

console.log("Extended Interface (WebDocument):", webDoc);
console.log(`WebDocument URL: ${webDoc.url}`);
console.log(`WebDocument CreatedAt: ${webDoc.createdAt}`);



// ==========================================================================
// 3. Declaration Merging (Unique to Interfaces)
// ==========================================================================
console.log("\n--- 3. Declaration Merging ---");

// If you declare an interface with the same name more than once *in the same scope*,
// TypeScript merges their definitions into a single interface.
// This is not possible with `type` aliases (duplicate `type` names cause errors).

interface User {
  id: string;
  name: string;
}

interface User {
  // This declaration MERGES with the one above
  email: string;
  preferredLanguage?: string; // Adding more properties
}

// The `User` interface now effectively has id, name, email, and preferredLanguage?
const mergedUser: User = {
  id: "user-merged-1",
  name: "Alice",
  email: "alice@example.com"
  // preferredLanguage is optional
};

console.log("Merged Interface User:", mergedUser);

// Why is merging useful?
// - Augmenting interfaces defined in external libraries (e.g., adding custom properties to global types).
// - Splitting complex interface definitions across multiple files (though often explicit extension is clearer).
// Why can it be confusing?
// - Properties might seem to appear "magically" if declarations are far apart or in different files.



// ==========================================================================
// 4. `interface` vs. `type` for Objects - Comparison
// ==========================================================================
console.log("\n--- 4. `interface` vs. `type` for Objects ---");

// Both can define object shapes with optional/readonly properties.

// Key Differences Recap:
// 1. Syntax: `interface X { ... }` vs `type Y = { ... };`
// 2. Extending: `interface extends ...` vs `type = TypeA & TypeB;`
// 3. Merging: Interfaces merge, type aliases do not (cause errors).
// 4. Flexibility: `type` can alias *any* type (primitives, unions, tuples...). `interface` is primarily for object shapes.

// Common Convention / When to Use Which:
// - Use `interface` when defining the shape of objects or classes, especially if you anticipate
//   they might be extended or implemented, or if you need declaration merging (e.g., library augmentation).
//   Many find interface syntax cleaner specifically for object shapes. Error messages can sometimes be clearer.
// - Use `type` when defining union types, intersection types (though can be used for objects too),
//   tuples, function types, primitive aliases, or utilizing more advanced features like mapped/conditional types.

// BEST PRACTICE: Choose one style for defining object shapes (`interface` or `type`) and be *consistent* within your project/team.
// A very common and recommended approach is:
//   - Use `interface` for defining object shapes.
//   - Use `type` for everything else (unions, intersections, primitives, tuples, function types, etc.).
// We will generally follow this convention.



// ==========================================================================
// 5. RAG Context Examples using `interface`
// ==========================================================================
console.log("\n--- 5. RAG Context Examples using `interface` ---");

// Let's redefine some RAG structures using interfaces.

interface RetrievalParams {
  query: string;
  topK: number;
  similarityThreshold?: number; // Optional
  targetSources?: string[];    // Optional
}

interface RetrievedChunk {
  readonly documentId: string;
  readonly chunkId: string;
  text: string;
  score: number;
  metadata?: object; // Using a generic object type for now, could be more specific
}

// Interface for the final context sent to the LLM
interface LLMContext {
  systemPrompt: string;
  userQuery: string;
  contextChunks: RetrievedChunk[]; // An array of objects matching the RetrievedChunk interface
}

const retrievalRequest: RetrievalParams = {
  query: "Explain TypeScript interfaces",
  topK: 3,
  targetSources: ["typescript-docs", "stack-overflow"]
};

const chunkForLLM: RetrievedChunk = {
  documentId: "ts-docs-interfaces",
  chunkId: "ts-docs-interfaces-chunk-2",
  text: "Interfaces can be extended using the `extends` keyword...",
  score: 0.91
};

const contextToSend: LLMContext = {
  systemPrompt: "You are a helpful assistant explaining technical concepts.",
  userQuery: retrievalRequest.query,
  contextChunks: [chunkForLLM] // Array containing our chunk object
};

console.log("Retrieval Parameters (using interface):", retrievalRequest);
console.log("Context Chunk (using interface):", chunkForLLM);
console.log("Context for LLM (using interface):", contextToSend);

// --- End of Module 5 ---