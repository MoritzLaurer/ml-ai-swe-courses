// 02-collections/arrays-and-tuples.ts

/**
 * Module 2: Working with Collections
 * Covers: Arrays (ordered lists, dynamic size) and Tuples (ordered lists, fixed size, specific types per position)
 */

console.log("--- Module 2: Working with Collections ---");



// ==========================================================================
// 1. Arrays
// ==========================================================================
console.log("\n--- 1. Arrays ---");

// Arrays represent ordered lists of values. Typically, they hold values of the same type.
// Similar to Python lists.

// Declaring and Initializing Arrays:
// Method 1: Using square brackets `[]` with a type annotation.
const documentIds: string[] = ["doc-abc-1", "doc-xyz-2", "doc-pqr-3"];

// Method 2: Using the generic `Array<T>` syntax. (Less common but equivalent)
const scores: Array<number> = [0.95, 0.88, 0.91];

// Method 3: Type inference (if initialized with values of the same type).
const keywords = ["RAG", "TypeScript", "LLM"]; // TypeScript infers this as string[]

console.log(`Document IDs (string[]):`, documentIds);
console.log(`Scores (Array<number>):`, scores);
console.log(`Keywords (inferred string[]):`, keywords);

// Accessing Elements (Zero-based index):
console.log(`First document ID: ${documentIds[0]}`); // Accesses "doc-abc-1"
console.log(`Second score: ${scores[1]}`);       // Accesses 0.88

// Array Properties and Common Methods:
// `.length`: Get the number of elements.
console.log(`Number of keywords: ${keywords.length}`); // Output: 3

// `.push()`: Add one or more elements to the *end* of the array (mutates the array).
keywords.push("VectorDB");
console.log(`Keywords after push:`, keywords); // Includes "VectorDB"

// `.pop()`: Remove the *last* element from the array and return it (mutates the array).
const removedKeyword = keywords.pop();
console.log(`Removed keyword: ${removedKeyword}`);
console.log(`Keywords after pop:`, keywords);

// Type Safety with Arrays:
// TypeScript prevents adding elements of the wrong type.
// scores.push("high"); // Uncommenting causes TS error: Argument of type 'string' is not assignable to parameter of type 'number'.

// Arrays can hold complex types too (we'll define interfaces/types properly later)
const initialDocuments: { id: string; score: number }[] = [
  { id: "doc-init-1", score: 0.7 },
  { id: "doc-init-2", score: 0.8 },
];
console.log(`Initial Documents (array of objects):`, initialDocuments);

// Iterating over arrays (briefly - more in Functions module):
console.log("Iterating through scores:");
scores.forEach((score, index) => {
  console.log(`  Score at index ${index}: ${score}`);
});

// `map` creates a new array by transforming each element (common pattern)
const formattedScores = scores.map(score => `Score: ${(score * 100).toFixed(1)}%`);
console.log(`Formatted Scores (using map):`, formattedScores);




// ==========================================================================
// 2. Tuples
// ==========================================================================
console.log("\n--- 2. Tuples ---");

// Tuples are like arrays but with a *fixed number* of elements, where the *type* of each element at a specific position is known and enforced.
// Similar to Python tuples, but with stricter type checking per element.

// Declaring and Initializing Tuples:
// Use square brackets with types listed in order for the annotation.
let documentIdAndScore: [string, number];

// Assign values matching the declared types and order.
documentIdAndScore = ["doc-final-xyz", 0.98];
console.log(`Document ID and Score Tuple:`, documentIdAndScore);

// Accessing Elements (Zero-based index, types are known):
const docId: string = documentIdAndScore[0];
const docScore: number = documentIdAndScore[1];
console.log(`Accessing tuple elements: ID=${docId}, Score=${docScore}`);

// Type Safety with Tuples:
// documentIdAndScore[0] = 123; // Error: Type 'number' is not assignable to type 'string'.
// documentIdAndScore[1] = "high"; // Error: Type 'string' is not assignable to type 'number'.
// documentIdAndScore[2] = "extra"; // Error: Tuple type '[string, number]' of length '2' has no element at index '2'.

// Size is fixed (mostly):
// While technically some array methods like `push` might work on tuples due to underlying JS representation,
// TypeScript strongly discourages this and provides errors when accessing elements beyond the defined length.
// Best practice: Treat tuples as fixed-size structures.
// documentIdAndScore.push("something else"); // This *might* not error immediately depending on TS version/config, but is bad practice and accessing index 2 will error.

// Common Use Cases for Tuples:
// - Representing fixed structures like coordinates: `[number, number]`
// - Representing key-value pairs: `[string, any]` (though objects/interfaces are often better)
// - Returning multiple values from a function (though returning an object is often clearer).

// Destructuring Tuples (common and convenient):
const [id, scoreVal] = documentIdAndScore;
console.log(`Destructured tuple: ID=${id}, Score=${scoreVal}`);

// Tuples with optional elements (?) - less common but possible
let coord3d: [number, number, number?]; // Z coordinate is optional
coord3d = [10, 20];
console.log(`Optional tuple element:`, coord3d);
coord3d = [5, 15, 7];
console.log(`Optional tuple element (set):`, coord3d);




// ==========================================================================
// 3. RAG Context Examples
// ==========================================================================
console.log("\n--- 3. RAG Context Examples ---");

// Array of relevant document IDs retrieved
const relevantDocIds: string[] = ["rag-ts-intro", "rag-use-cases", "rag-challenges"];
console.log(`Relevant Doc IDs:`, relevantDocIds);

// Array of scores corresponding to some documents
const relevanceScores: number[] = [0.92, 0.85, 0.78, 0.91];
console.log(`Relevance Scores:`, relevanceScores);

// Array of tuples, each containing a document ID and its score
const rankedDocuments: [string, number][] = [
  ["rag-ts-intro", 0.92],
  ["rag-challenges", 0.91], // Note order might differ from relevanceScores example
  ["rag-use-cases", 0.85]
];
console.log(`Ranked Documents (Array of Tuples):`, rankedDocuments);

// Accessing data from the array of tuples
console.log(`Top ranked document ID: ${rankedDocuments[0][0]}`);
console.log(`Top ranked document score: ${rankedDocuments[0][1]}`);

// Using destructuring in a loop (preview for later)
console.log("Processing ranked documents:");
rankedDocuments.forEach(([docId, score]) => {
  console.log(`  Processing ${docId} with score ${score}`);
});

// --- End of Module 2 ---