// 07-control-flow/control-flow.ts

/**
 * Module 7: Making Decisions & Repeating Actions - Control Flow
 * Covers: if/else if/else, switch, ternary operator, for loops (classic, for...of, for...in),
 * while/do...while loops, break/continue, truthiness/falsiness.
 */

// Simple type for RAG context examples
type RetrievedDoc = { id: string; score: number; text: string; };

console.log("--- Module 7: Control Flow ---");



// ==========================================================================
// 1. Conditional Statements: if / else if / else
// ==========================================================================
console.log("\n--- 1. Conditional Statements ---");

const score: number = 0.85;
const threshold: number = 0.75;
const minScore: number = 0.5;

if (score >= threshold) {
  console.log(`Score ${score} meets or exceeds threshold ${threshold}. Highly relevant.`);
} else if (score >= minScore) {
  console.log(`Score ${score} is below threshold ${threshold} but above minimum ${minScore}. Moderately relevant.`);
} else {
  console.log(`Score ${score} is below minimum ${minScore}. Not relevant.`);
}

// --- Truthiness and Falsiness ---
// In conditional contexts (like `if`), values are coerced to boolean.
// Values that evaluate to `false` are called "falsy":
//   - false
//   - 0 and -0
//   - 0n (BigInt zero)
//   - "" (empty string)
//   - null
//   - undefined
//   - NaN (Not a Number)
// ALL other values are "truthy", including:
//   - true
//   - any non-zero number (e.g., 1, -1, 0.5)
//   - any non-empty string (e.g., "hello", "0", "false")
//   - empty arrays `[]` (DIFFERENT from Python!)
//   - empty objects `{}` (DIFFERENT from Python!)
//   - functions

console.log("\n--- Truthiness/Falsiness Examples ---");
// @ts-ignore - Demonstrating truthiness with a non-empty string
if ("hello") { console.log("'hello' is truthy"); }
// @ts-ignore - Demonstrating truthiness with a non-zero number
if (10) { console.log("10 is truthy"); }
// @ts-ignore - Demonstrating truthiness with an empty array
if ([]) { console.log("[] (empty array) is truthy in JS/TS"); }
// @ts-ignore - Demonstrating truthiness with an empty object
if ({}) { console.log("{} (empty object) is truthy in JS/TS"); }

if (0) { /* This block won't run */ } else { console.log("0 is falsy"); }
// @ts-ignore - Demonstrating falsiness with an empty string
if ("") { /* This block won't run */ } else { console.log("'' (empty string) is falsy"); }
// @ts-ignore - Demonstrating falsiness with null
if (null) { /* This block won't run */ } else { console.log("null is falsy"); }
// @ts-ignore - Demonstrating falsiness with undefined
if (undefined) { /* This block won't run */ } else { console.log("undefined is falsy"); }
if (NaN) { /* This block won't run */ } else { console.log("NaN is falsy"); }



// ==========================================================================
// 2. Ternary Operator (Conditional Expression)
// ==========================================================================
console.log("\n--- 2. Ternary Operator ---");

// A concise way to write simple if/else assignments or expressions.
// Syntax: condition ? valueIfTrue : valueIfFalse

const relevanceStatus = score >= threshold ? "High" : "Low";
console.log(`Relevance Status (Ternary): ${relevanceStatus}`);

const message = score > 0.9 ? "Excellent match!" : score > 0.6 ? "Good match" : "Fair match"; // Can be chained (use parentheses for clarity if needed)
console.log(`Match message (Chained Ternary): ${message}`);



// ==========================================================================
// 3. `switch` Statement
// ==========================================================================
console.log("\n--- 3. `switch` Statement ---");

// Used for evaluating an expression against multiple possible constant values (cases).

type RagStatus = "initializing" | "retrieving" | "generating" | "complete" | "error";
const currentStatus: RagStatus = "generating";

switch (currentStatus) {
  // @ts-ignore - Case type comparison for educational switch example
  case "initializing":
    console.log("System is starting up...");
    break; // IMPORTANT: `break` exits the switch. Without it, execution "falls through" to the next case.
  // @ts-ignore - Case type comparison for educational switch example
  case "retrieving":
    console.log("Finding relevant documents...");
    break;
  case "generating":
    console.log("Creating response based on context...");
    // Fall-through example (sometimes intentional, often accidental):
    // If we forget break here, it will also execute the 'complete' block!
    break; // Added break
  // @ts-ignore - Case type comparison for educational switch example
  case "complete":
    console.log("Process finished successfully.");
    break;
  // @ts-ignore - Case type comparison for educational switch example
  case "error":
    console.error("An error occurred during the process.");
    break;
  default:
    // Optional: Executes if no other case matches.
    console.warn(`Unknown status encountered: ${currentStatus}`);
    break;
}



// ==========================================================================
// 4. Loops: `for`, `while`, `do...while`
// ==========================================================================
console.log("\n--- 4. Loops ---");

// Classic `for` loop (mostly used for indexed iteration, not for arrays or objects)
// for arrays, use `for...of` or `for...in`; for objects, use `for...in` (see further below)
console.log("Classic for loop:");
for (let i: number = 0; i < 3; i++) { // Initialize; Condition; Increment
  console.log(`  Iteration ${i}`);
}

// `while` loop (executes as long as the condition is true)
console.log("while loop:");
let countdown: number = 3;
while (countdown > 0) {
  console.log(`  Countdown: ${countdown}`);
  countdown--; // Decrement the counter
}
console.log("  Blast off!");

// `do...while` loop (executes the block *once* first, then checks the condition)
console.log("do...while loop:");
let attempt = 0;
let success = false;
do {
  attempt++;
  console.log(`  Attempt ${attempt}...`);
  // Simulate a process that might succeed
  if (attempt === 2) {
    success = true;
    console.log("  Success!");
  }
} while (!success && attempt < 3);



// ==========================================================================
// 5. Iterating with `for...of` (for Arrays and other Iterables)
// ==========================================================================
console.log("\n--- 5. `for...of` Loop ---");

// The PREFERRED way to iterate over the *values* of an iterable (Arrays, Strings, Maps, Sets, etc.).
// Similar to Python's `for item in list:`.

const docIds: string[] = ["doc-A", "doc-B", "doc-C"];
console.log("Iterating over docIds array:");
for (const id of docIds) {
  console.log(`  Processing ID: ${id}`);
  // `id` directly holds the value ("doc-A", then "doc-B", etc.)
}

const messageString = "RAG";
console.log("Iterating over messageString:");
for (const char of messageString) {
  console.log(`  Character: ${char}`); // R, then A, then G
}



// ==========================================================================
// 6. Iterating with `for...in` (for Object Keys - Use with Caution!)
// ==========================================================================
console.log("\n--- 6. `for...in` Loop ---");

// Iterates over the *keys* (property names) of an object.
// CAUTION:
//   - Iterates over enumerable properties, including inherited ones (unless checked with `hasOwnProperty`).
//   - The order of iteration is not guaranteed.
//   - The key is always a string.

const configObject = { model: "Model-X", temperature: 0.7, topK: 5 };
console.log("Iterating over configObject keys with for...in:");
for (const key in configObject) {
  // It's good practice to check if the property belongs directly to the object
  if (Object.prototype.hasOwnProperty.call(configObject, key)) {
    // To get the value, use bracket notation:
    // We need type assertion here or better type handling, TS doesn't know the type of configObject[key] easily.
    const value = (configObject as any)[key];
    console.log(`  Key: ${key}, Value: ${value}`);
  }
}

// BETTER ways to iterate over objects (with for...of):
console.log("Better object iteration (Object.keys):");
for (const key of Object.keys(configObject)) {
  const value = configObject[key as keyof typeof configObject]; // More type-safe access
  console.log(`  Key: ${key}, Value: ${value}`);
}

console.log("Better object iteration (Object.entries):");
for (const [key, value] of Object.entries(configObject)) {
  // `key` is string, `value` has the correct inferred type here
  console.log(`  Key: ${key}, Value: ${value}`);
}
// BEST PRACTICE: Prefer `Object.keys()`, `Object.values()`, or `Object.entries()` with `for...of` over `for...in` for objects.



// ==========================================================================
// 7. `break` and `continue`
// ==========================================================================
console.log("\n--- 7. `break` and `continue` ---");

// `break`: Exits the current loop (`for`, `while`, `do...while`) or `switch` statement entirely.
console.log("Loop with break:");
for (let i = 0; i < 10; i++) {
  if (i === 3) {
    console.log("  Breaking loop at i=3");
    break; // Stop the loop
  }
  console.log(`  i = ${i}`);
} // Output stops after i=2

// `continue`: Skips the rest of the current iteration and proceeds to the next one.
console.log("Loop with continue:");
for (let i = 0; i < 5; i++) {
  if (i === 2) {
    console.log("  Continuing loop at i=2 (skipping console.log)");
    continue; // Skip the console.log for this iteration
  }
  console.log(`  i = ${i}`);
} // Output skips i=2



// ==========================================================================
// 8. RAG Context Examples
// ==========================================================================
console.log("\n--- 8. RAG Context Examples ---");

const retrievedDocs: RetrievedDoc[] = [
  { id: "doc-1", score: 0.92, text: "Content about TS..." },
  { id: "doc-2", score: 0.65, text: "Content about JS..." }, // Below threshold
  { id: "doc-3", score: 0.88, text: "More TS content..." },
  { id: "doc-4", score: 0.78, text: "Context on RAG..." },
];
const relevanceThreshold = 0.75;
const maxDocsToUse = 2;
const contextForLLM: string[] = []; // Array to hold relevant text

console.log(`Filtering and formatting documents (threshold=${relevanceThreshold}, max=${maxDocsToUse}):`);
for (const doc of retrievedDocs) {
  console.log(`  Checking doc: ${doc.id} (Score: ${doc.score})`);
  if (doc.score < relevanceThreshold) {
    console.log(`    Score below threshold, skipping.`);
    continue; // Skip to the next document
  }

  if (contextForLLM.length >= maxDocsToUse) {
    console.log(`    Reached max documents limit (${maxDocsToUse}), stopping.`);
    break; // Exit the loop
  }

  console.log(`    Score meets threshold, adding to context.`);
  contextForLLM.push(`[${doc.id}]: ${doc.text}`);
}

console.log("\nFinal Context for LLM:");
if (contextForLLM.length > 0) {
  contextForLLM.forEach((text, index) => console.log(`${index + 1}. ${text.substring(0, 50)}...`)); // Print truncated context
} else {
  console.log("No relevant documents found or added.");
}

// --- End of Module 7 ---