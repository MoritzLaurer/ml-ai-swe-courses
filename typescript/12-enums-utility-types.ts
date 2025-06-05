// 12-enums-utility-types/enums-utils.ts

/**
 * Module 12: Enums & Utility Types
 *
 * Purpose: This script demonstrates two important TypeScript features:
 * 1. Enums (`enum`): For creating sets of named constants (numeric or string-based),
 *    improving code clarity and maintainability compared to using raw numbers or strings.
 * 2. Utility Types: Built-in generic types (like Partial, Readonly, Pick, Omit, Record)
 *    that allow transforming existing types into new ones based on common patterns,
 *    reducing boilerplate code and enhancing type safety.
 *
 * The examples showcase how to define and use enums, and how various utility types
 * can manipulate interfaces, particularly using a hypothetical `RAGConfig` as a base.
 */

console.log("--- Module 12: Enums & Utility Types ---");



// ==========================================================================
// 1. Enums: Named Constants
// ==========================================================================
console.log("\n--- 1. Enums ---");

// Enums allow defining a set of named constants, improving code clarity over magic numbers/strings.

// --- Numeric Enums ---
// By default, enums assign incrementing numbers starting from 0.
enum Direction {
  Up,    // 0
  Down,  // 1
  Left,  // 2
  Right  // 3
}

let move: Direction = Direction.Up;
console.log(`Numeric Enum - Initial Move: ${move}`); // Output: 0 (the numeric value)
console.log(`Numeric Enum - Member Name: ${Direction[move]}`); // Output: Up (Reverse Mapping!)

// You can explicitly set the starting number or values for specific members.
enum ResponseStatus {
  Success = 200,
  BadRequest = 400,
  Unauthorized = 401,
  NotFound = 404,
  ServerError = 500
}

let statusCheck: ResponseStatus = ResponseStatus.NotFound;
console.log(`Numeric Enum (assigned) - Status: ${statusCheck}`); // Output: 404
console.log(`Numeric Enum (assigned) - Status Name: ${ResponseStatus[404]}`); // Output: NotFound

// Reverse Mapping Caveat: Numeric enums create a reverse mapping from value to name in the compiled JS.
// This can sometimes be useful but also adds code size.

// --- String Enums ---
// Members are initialized with string literals. Generally preferred for readability and debugging.
// No reverse mapping is generated.
enum LogLevel {
  DEBUG = "DEBUG",
  INFO = "INFO",
  WARN = "WARN",
  ERROR = "ERROR"
}

let appLogLevel: LogLevel = LogLevel.INFO;
console.log(`String Enum - Log Level: ${appLogLevel}`); // Output: INFO
// console.log(LogLevel["INFO"]); // This doesn't work directly like reverse mapping

function logMessage(level: LogLevel, message: string): void {
  // Can use enums in function parameters for type safety
  console.log(`[${level}] ${message}`);
}
logMessage(LogLevel.WARN, "Configuration value might be suboptimal.");

// --- Const Enums ---
// `const enum Size { Small, Medium, Large }`
// These are completely removed during compilation, and their values are inlined.
// Benefits: Better performance (no runtime lookup).
// Drawbacks: Cannot access via computed property names (e.g., Size[computedValue]), no reverse mapping.
// Recommendation: Use regular enums unless you specifically need the inlining and understand the limitations.



// ==========================================================================
// 2. Introduction to Utility Types
// ==========================================================================
// Utility Types are built-in generic types that help transform existing types
// into new ones (e.g., making properties optional, picking a subset).
// They reduce boilerplate and enhance type safety for common transformations.
console.log("\n--- 2. Introduction to Utility Types ---");

// TypeScript provides built-in generic types (Utility Types) to help manipulate existing types.
// They make common type transformations easier without defining new types manually.
// This section primarily sets up the 'RAGConfig' interface and 'baseConfig' object,
// which will be used as the basis for demonstrating various utility types in the following sections.
// No specific utility type is applied here yet.

// Let's define a base interface to work with:
interface RAGConfig {
  model: string;
  temperature: number;
  topK: number;
  retrievalSource: string;
  systemPrompt?: string; // Optional property
}

const baseConfig: RAGConfig = {
  model: "rag-model-v4",
  temperature: 0.7,
  topK: 5,
  retrievalSource: "internal_docs"
  // systemPrompt is omitted (optional)
};



// ==========================================================================
// 3. Partial<T> and Required<T>
// ==========================================================================
console.log("\n--- 3. Partial<T> and Required<T> ---");

// `Partial<T>`: Makes all properties of T optional.
// Useful for objects representing updates where only some fields are provided.

type PartialRAGConfig = Partial<RAGConfig>;
// Equivalent to:
// type PartialRAGConfig = {
//   model?: string;
//   temperature?: number;
//   topK?: number;
//   retrievalSource?: string;
//   systemPrompt?: string; // Still optional
// }

const configUpdate: PartialRAGConfig = {
  temperature: 0.5, // Update only temperature
  topK: 3          // and topK
};
console.log("Partial Config Update:", configUpdate);

// Function that accepts partial updates
function applyUpdate(currentConfig: RAGConfig, update: PartialRAGConfig): RAGConfig {
  // Use spread syntax to merge (update overrides currentConfig)
  return { ...currentConfig, ...update };
}
const updatedConfig = applyUpdate(baseConfig, configUpdate);
console.log("Config after Partial Update:", updatedConfig);

// `Required<T>`: Makes all properties of T required (removes `?`).
type RequiredRAGConfig = Required<RAGConfig>;
// Equivalent to:
// type RequiredRAGConfig = {
//   model: string;
//   temperature: number;
//   topK: number;
//   retrievalSource: string;
//   systemPrompt: string; // Now required!
// }

// const incompleteRequired: RequiredRAGConfig = { ...baseConfig }; // Error: Property 'systemPrompt' is missing...



// ==========================================================================
// 4. Readonly<T>
// ==========================================================================
console.log("\n--- 4. Readonly<T> ---");

// `Readonly<T>`: Makes all properties of T readonly.
// Useful for creating immutable versions of objects.

type ImmutableConfig = Readonly<RAGConfig>;
// Equivalent to:
// type ImmutableConfig = {
//   readonly model: string;
//   readonly temperature: number;
//   readonly topK: number;
//   readonly retrievalSource: string;
//   readonly systemPrompt?: string; // Optionality preserved, but becomes readonly if present
// }

const frozenConfig: ImmutableConfig = { ...baseConfig, systemPrompt: "Be concise." };
console.log("Readonly Config:", frozenConfig);

// frozenConfig.model = "new-model"; // Error: Cannot assign to 'model' because it is a read-only property.
// frozenConfig.systemPrompt = "Be verbose"; // Error: Cannot assign to 'systemPrompt'...



// ==========================================================================
// 5. Pick<T, K> and Omit<T, K>
// ==========================================================================
console.log("\n--- 5. Pick<T, K> and Omit<T, K> ---");

// `Pick<T, K>`: Creates a new type by selecting only specific properties `K` from type `T`.
// `K` must be a string literal or a union of string literals that are keys of `T`.
type ModelInfo = Pick<RAGConfig, "model" | "temperature">;
// Equivalent to: type ModelInfo = { model: string; temperature: number; }

const modelDetails: ModelInfo = {
  model: baseConfig.model,
  temperature: baseConfig.temperature
};
console.log("Picked Properties (ModelInfo):", modelDetails);

// `Omit<T, K>`: Creates a new type by taking all properties from `T` and removing properties `K`.
type RetrievalConfig = Omit<RAGConfig, "model" | "temperature" | "systemPrompt">;
// Equivalent to: type RetrievalConfig = { topK: number; retrievalSource: string; }

const retrievalSettings: RetrievalConfig = {
  topK: baseConfig.topK,
  retrievalSource: baseConfig.retrievalSource
};
console.log("Omitted Properties (RetrievalConfig):", retrievalSettings);



// ==========================================================================
// 6. Record<K, T>
// ==========================================================================
console.log("\n--- 6. Record<K, T> ---");

// `Record<K, T>`: Constructs an object type where keys are of type `K` and values are of type `T`.
// `K` is usually `string | number | symbol`. Useful for dictionaries/maps with known value types.

type UserScores = Record<string, number>; // Keys are strings (user IDs), values are numbers (scores)

const scores: UserScores = {
  "user-1": 0.95,
  "user-2": 0.88,
  "user-3": 0.75
  // "user-4": "high" // Error: Type 'string' is not assignable to type 'number'.
};
console.log("Record Type (UserScores):", scores);

type FeatureFlags = Record<string, boolean | undefined>; // Allow boolean flags, possibly undefined
const flags: FeatureFlags = {
  "useCache": true,
  "enableDebugLog": false,
  "newSearchUI": undefined // Explicitly undefined is allowed
};
console.log("Record Type (FeatureFlags):", flags);



// ==========================================================================
// 7. ReturnType<T> and Parameters<T>
// ==========================================================================
console.log("\n--- 7. ReturnType<T> and Parameters<T> ---");

// These utility types work on function types.

// `ReturnType<T>`: Obtains the return type of a function type `T`.
type ConfigUpdateFunction = (cfg: RAGConfig, upd: PartialRAGConfig) => RAGConfig;
type ConfigUpdateResult = ReturnType<ConfigUpdateFunction>; // Result is RAGConfig
const resultOfUpdate: ConfigUpdateResult = applyUpdate(baseConfig, {}); // Type checks!

// `Parameters<T>`: Obtains the parameter types of a function type `T` as a tuple type.
type ConfigUpdateParams = Parameters<ConfigUpdateFunction>; // Result is [RAGConfig, PartialRAGConfig]
const paramsForUpdate: ConfigUpdateParams = [baseConfig, { topK: 10 }]; // Type checks!



// ==========================================================================
// 8. RAG Context Examples
// ==========================================================================
console.log("\n--- 8. RAG Context Examples ---");

// Using an Enum for RAG System Status
enum RagProcessingStatus {
  Idle = "IDLE",
  ReceivingQuery = "RECEIVING_QUERY",
  RetrievingDocs = "RETRIEVING_DOCS",
  FormattingContext = "FORMATTING_CONTEXT",
  GeneratingResponse = "GENERATING_RESPONSE",
  Complete = "COMPLETE",
  Failed = "FAILED"
}

let currentRagState: RagProcessingStatus = RagProcessingStatus.Idle;
console.log(`Initial RAG State: ${currentRagState}`); // IDLE

function updateRagState(newState: RagProcessingStatus): void {
  currentRagState = newState;
  console.log(`RAG State changed to: ${currentRagState}`);
}
updateRagState(RagProcessingStatus.ReceivingQuery);
updateRagState(RagProcessingStatus.RetrievingDocs);

// Using Utility Types for document processing pipeline stages
interface FullDocument {
  id: string;
  rawText: string;
  metadata: object;
  chunks?: string[];
  vectors?: number[][];
  summary?: string;
}

// Type for just the initial ingestion stage
type IngestedDocument = Pick<FullDocument, "id" | "rawText" | "metadata">;
// Type for the chunking stage result
type ChunkedDocument = Required<Pick<FullDocument, "id" | "chunks">>;
// Type for the vectorization result (making vectors readonly)
type VectorizedDocument = Required<Pick<FullDocument, "id">> & { readonly vectors: number[][] };

const ingested: IngestedDocument = { id: 'doc-xyz', rawText: '...', metadata: { source: 'web' } };
const chunked: ChunkedDocument = { id: 'doc-xyz', chunks: ['chunk1...', 'chunk2...'] };
const vectorized: VectorizedDocument = { id: 'doc-xyz', vectors: [[0.1, 0.2], [0.3, 0.4]] };

console.log("Ingested Doc Shape:", ingested);
console.log("Chunked Doc Shape:", chunked);
console.log("Vectorized Doc Shape:", vectorized);

// --- End of Module 12 ---