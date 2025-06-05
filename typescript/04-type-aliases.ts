// 04-type-aliases/type-aliases.ts

/**
 * Module 4: Type Aliases
 * Covers: Using the `type` keyword to create custom names for type definitions,
 * focusing on object shapes, optional/readonly properties, unions, and intersections.
 */

console.log("--- Module 4: Type Aliases ---");

// In Module 3, we saw JavaScript objects. Now, we need a way to tell TypeScript
// exactly what properties (keys) and value types we expect an object to have.
// `type` aliases are one powerful way to do this.



// ==========================================================================
// 1. Basic `type` Aliases for Primitives and Objects
// ==========================================================================
console.log("\n--- 1. Basic `type` Aliases ---");

// Syntax: type NewName = TypeDefinition;

// Aliasing primitive types improves readability and expresses intent.
type UserID = string;
type DocumentID = string;
type ConfidenceScore = number;

let currentUserId: UserID = "rag-user-1138";
let activeDocId: DocumentID = "ts-intro-doc-v2";

console.log(`UserID: ${currentUserId}, DocumentID: ${activeDocId}`);
// Remember: These are just names (aliases), not distinct new types. UserID is still a string.

// Aliasing an object's shape - defining the structure we expect.
// This makes our code predictable and prevents errors like typos or missing properties.
type SimpleDocument = {
  id: DocumentID; // Use our alias
  title: string;
  contentLength: number;
  isVectorized: boolean;
};

// Now we can use 'SimpleDocument' as the type for our document objects.
const docA: SimpleDocument = {
  id: "doc-a-001",
  title: "Getting Started with TypeScript",
  contentLength: 5280,
  isVectorized: true
};

console.log("Document Object Typed with Alias:", docA);

// TypeScript enforces the shape defined by the type alias:
// const docB: SimpleDocument = { // Error: Property 'isVectorized' is missing...
//   id: "doc-b-002",
//   title: "JavaScript Objects",
//   contentLength: 3100
// };
// const docC: SimpleDocument = { // Error: Object literal may only specify known properties... 'year' does not exist...
//   id: "doc-c-003",
//   title: "Node.js Fundamentals",
//   contentLength: 4500,
//   isVectorized: false,
//   year: 2025 // <-- Extra property not allowed by the 'SimpleDocument' type
// };



// ==========================================================================
// 2. Optional (`?`) and Readonly (`readonly`) Properties
// ==========================================================================
console.log("\n--- 2. Optional and Readonly Properties ---");

// We can refine our object shapes further using modifiers.

type DocumentMetadata = {
  readonly sourceId: string; // Cannot be changed after object creation. Good for IDs.
  sourceSystem: string;
  ingestedAt: Date;
  page?: number;           // `?` means this property is optional (it can be `number` or `undefined`).
  tags?: string[];         // Optional array of strings.
  editorComment?: string;  // Another optional property.
};

// Example usage:
const metadata1: DocumentMetadata = {
  sourceId: "metadata-xyz-987",
  sourceSystem: "Main Content DB",
  ingestedAt: new Date(),
  // page, tags, and editorComment are not provided, which is valid.
};

const metadata2: DocumentMetadata = {
  sourceId: "metadata-abc-123",
  sourceSystem: "Archival System",
  ingestedAt: new Date(2024, 10, 15), // Month is 0-indexed (10 = November)
  page: 42,
  tags: ["archive", "legacy"]
  // editorComment is still optional and omitted here.
};

console.log("Metadata (Optional props omitted):", metadata1);
console.log("Metadata (Some optional props included):", metadata2);

// Accessing optional properties:
// If you access an optional property that wasn't provided, its value is `undefined`.
console.log(`metadata1.page: ${metadata1.page}`); // undefined
console.log(`metadata2.page: ${metadata2.page}`); // 42

// Trying to modify a readonly property gives a compile-time error:
// metadata1.sourceId = "new-id-attempt"; // Error: Cannot assign to 'sourceId' because it is a read-only property.

// BEST PRACTICE: Use `readonly` for properties that represent stable identifiers or configuration
// that shouldn't change once an object is created. Use `?` for data that may genuinely be absent.


// ==========================================================================
// 3. Union (`|`) and Intersection (`&`) Types
// ==========================================================================
console.log("\n--- 3. Union and Intersection Types ---");

// `type` aliases are excellent for defining union and intersection types.

// Union (`|`): Represents a value that can be one of several types.
type ID = string | number; // An ID can be either a string or a number

let id1: ID = "user-123";
let id2: ID = 456;
console.log(`Union Type ID (string): ${id1}, (number): ${id2}`);
// let id3: ID = false; // Error: Type 'boolean' is not assignable to type 'string | number'.

// Literal Union Types: Constrain a variable to a specific set of literal values.
type RetrievalStatus = "Idle" | "Fetching" | "Processing" | "Complete" | "Error";
let currentRagStatus: RetrievalStatus = "Idle";
console.log(`Literal Union Status: ${currentRagStatus}`);
currentRagStatus = "Complete";
// currentRagStatus = "Failed"; // Error: Type '"Failed"' is not assignable to type 'RetrievalStatus'.

// Intersection (`&`): Combines multiple types into a single type that has *all*
// the properties of the combined types.

type BaseResult = {
  timestamp: Date;
  queryId: string;
};

type SuccessfulRetrieval = BaseResult & { // Combine BaseResult properties...
  status: "Complete"; // ...with specific success properties
  retrievedDocs: DocumentID[];
  scores: ConfidenceScore[];
};

type FailedRetrieval = BaseResult & { // Combine BaseResult properties...
  status: "Error"; // ...with specific error properties
  errorCode: number;
  errorMessage: string;
};

// A variable can be typed as a union of these intersection types:
type RetrievalOutcome = SuccessfulRetrieval | FailedRetrieval;

// Example of a successful outcome:
const successOutcome: RetrievalOutcome = {
  timestamp: new Date(),
  queryId: "q-abc",
  status: "Complete", // Must match 'SuccessfulRetrieval'
  retrievedDocs: ["doc-1", "doc-2"],
  scores: [0.9, 0.8]
};

// Example of a failed outcome:
const failureOutcome: RetrievalOutcome = {
  timestamp: new Date(),
  queryId: "q-xyz",
  status: "Error", // Must match 'FailedRetrieval'
  errorCode: 503,
  errorMessage: "Retrieval system timeout"
};

console.log("Successful Retrieval Outcome:", successOutcome);
console.log("Failed Retrieval Outcome:", failureOutcome);

// This pattern (base type + status-specific types combined with unions) is very common for representing API responses or state.



// ==========================================================================
// 4. RAG Context Examples using `type`
// ==========================================================================
console.log("\n--- 4. RAG Context Examples ---");

// Define a type for the overall RAG system configuration
type RAGSystemConfig = {
  readonly systemId: string;
  model: string;
  embeddingModel: string;
  retrievalTopK: number;
  responseMode: "concise" | "detailed"; // Literal union
  logLevel?: "debug" | "info" | "warn" | "error"; // Optional literal union
  featureFlags: { // Nested object shape defined inline
    useHybridSearch: boolean;
    enableStreaming?: boolean; // Optional flag
  };
};

const config: RAGSystemConfig = {
  systemId: "rag-prod-us-east-1",
  model: "super-rag-model-v3",
  embeddingModel: "text-embedding-ada-002",
  retrievalTopK: 5,
  responseMode: "detailed",
  logLevel: "info",
  featureFlags: {
    useHybridSearch: true,
    enableStreaming: false
  }
};

console.log("RAG System Configuration:", config);

// Define a type for a single processed document chunk ready for the LLM
type ContextChunk = {
  readonly documentId: DocumentID;
  chunkId: string;
  text: string;
  score: ConfidenceScore;
  metadata?: DocumentMetadata; // Optional metadata using our previous type alias
};

const chunk1: ContextChunk = {
  documentId: "ts-intro-doc-v2",
  chunkId: "ts-intro-chunk-3",
  text: "...TypeScript adds optional types to JavaScript that support tools for large-scale JavaScript applications...",
  score: 0.95,
  metadata: { // Include optional metadata
    sourceId: "ts-homepage-meta",
    sourceSystem: "Web Scraper",
    ingestedAt: new Date(),
    tags: ["typescript", "javascript", "static-typing"]
  }
};

console.log("Processed Context Chunk:", chunk1);

// --- End of Module 4 ---