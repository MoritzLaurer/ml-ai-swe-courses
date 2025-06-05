// 01-basics/variables-primitives-operators.ts

/**
 * Module 1: TypeScript Fundamentals
 * Covers: Variable declaration, Primitive Types, Type Annotations & Inference, Basic Operators
 */

console.log("--- Module 1: TypeScript Fundamentals ---");



// ==========================================================================
// 1. Variable Declaration: let vs const
// ==========================================================================
console.log("\n--- 1. Variable Declaration ---");

// Use `let` to declare variables whose values might change later (mutable binding).
// Similar to a standard variable assignment in Python.
let userQuery: string = "What is Retrieval-Augmented Generation?";
console.log(`Initial Query (let): ${userQuery}`);

userQuery = "Explain RAG in simple terms"; // Allowed, because `userQuery` was declared with `let`.
console.log(`Updated Query (let): ${userQuery}`);

// Use `const` to declare variables whose assignment will *not* change (immutable binding).
// Once a value is assigned to a const, it cannot be reassigned.
// This is generally preferred for variables that don't need to be reassigned, making code safer and easier to understand.
// Analogy: conceptually like constants, but specifically prevents reassignment.
const maxDocuments = 5;
console.log(`Max Documents (const): ${maxDocuments}`);

// maxDocuments = 10; // Uncommenting this line causes a TypeScript error:
                   // "Cannot assign to 'maxDocuments' because it is a constant."

// IMPORTANT NOTE for `const` with objects/arrays:
// `const` makes the *binding* immutable, not necessarily the *value* itself if it's an object or array.
const config = { model: "gpt-4", temperature: 0.7 };
config.temperature = 0.5; // This IS allowed because we're changing a property *inside* the object, not reassigning `config`.
// config = { model: "claude-3", temperature: 0.6 }; // This IS NOT allowed (reassignment).
console.log(`Config object (const, but mutable content):`, config);

// BEST PRACTICE: Default to using `const` and only use `let` when you know you need to reassign the variable.



// ==========================================================================
// 2. Primitive Types
// ==========================================================================
console.log("\n--- 2. Primitive Types ---");

// TypeScript uses JavaScript's primitive types.

// string: Textual data (single quotes, double quotes, or backticks for template literals)
const botName: string = 'RAG Assistant'; // Explicit type annotation
let greeting = `Hello from ${botName}!`; // Type 'string' inferred
console.log(`String: ${greeting}`);

// number: Represents both integers and floating-point numbers. No separate int/float.
const documentsRetrieved: number = 3; // Integer
let relevanceScore: number = 0.895; // Floating point
console.log(`Number (integer): ${documentsRetrieved}, Number (float): ${relevanceScore}`);

// boolean: Represents true or false values.
const isEnabled: boolean = true;
let needsContext = false; // Type 'boolean' inferred
console.log(`Boolean: ${isEnabled}, ${needsContext}`);

// null: Represents the intentional absence of an object value. Often used explicitly by programmers.
// In Python, this is analogous to `None`.
let llmResponse: string | null = null; // We might not have a response yet. Using a Union Type (more later)
console.log(`Null: ${llmResponse}`);
// llmResponse = "Here is the generated answer..."; // Now it has a value

// undefined: Represents a value that is not defined or assigned. Often used implicitly by JavaScript/TypeScript.
// - Variables declared but not initialized are `undefined`.
// - Function parameters not provided are `undefined`.
// - Object properties that don't exist yield `undefined`.
// Python doesn't have a direct equivalent; `None` is used more broadly, but `undefined` is distinct.
let userSessionId; // Declared but not initialized, its value is undefined
console.log(`Undefined (uninitialized variable): ${userSessionId}`);

// Difference between null and undefined:
// `null` is an assigned value meaning "no value" (intentional absence).
// `undefined` typically means a variable hasn't been assigned a value yet, or something doesn't exist.
// While sometimes used interchangeably, it's good practice to use `null` for intentional absence.

// bigint: For integers larger than the maximum safe integer representable by `number`. Use 'n' suffix.
// Not typically needed unless dealing with extremely large numbers.
// const veryLargeNumber: bigint = 9007199254740991n * 2n;
// console.log(`BigInt: ${veryLargeNumber}`);

// symbol: Represents unique identifiers. Less common in everyday application code.
// const uniqueKey = Symbol("userId");
// console.log(`Symbol: ${uniqueKey.toString()}`);


// ==========================================================================
// 3. Type Annotations vs. Type Inference
// ==========================================================================
console.log("\n--- 3. Type Annotations vs. Type Inference ---");

// Type Annotation (Explicit): We explicitly tell TypeScript the type.
let apiKey: string = "xyz-123-abc"; // We wrote `: string`

// Type Inference (Implicit): TypeScript automatically figures out the type based on the assigned value.
let requestTimestamp = new Date(); // TS infers `requestTimestamp` is of type `Date`
let defaultModel = "gpt-3.5-turbo"; // TS infers `defaultModel` is of type `string`

console.log(`Inferred type for timestamp: ${requestTimestamp.toISOString()}`);
console.log(`Inferred type for model: ${defaultModel}`);

// When to use explicit annotations?
// 1. Declaring a variable without initializing it:
let currentStatus: string; // Annotation needed, no initial value to infer from.
// currentStatus = "Processing"; // Now assigned
// console.log(currentStatus);

// 2. Function parameters and return types (CRUCIAL!): We'll cover this in the Functions module.
// function processQuery(query: string): boolean { /* ... */ return true; }

// 3. When you want to be more specific than inference allows (e.g., union types, though TS often infers these well too).
let userId: string | number = "user-456"; // Explicitly allow string OR number
userId = 789; // Also allowed

// BEST PRACTICE: Rely on type inference when the type is obvious from the initial value (like `let name = "Alice"`).
// Use explicit annotations for function signatures, uninitialized variables, and complex types.


// ==========================================================================
// 4. Basic Operators
// ==========================================================================
console.log("\n--- 4. Basic Operators ---");

// Arithmetic Operators: +, -, *, /, % (modulo), ** (exponentiation)
let numDocs = 5;
let score = 0.8;
let totalScore = numDocs * score;
console.log(`Arithmetic: ${numDocs} * ${score} = ${totalScore}`);
console.log(`Arithmetic (modulo): 10 % 3 = ${10 % 3}`);
console.log(`Arithmetic (exponent): 2 ** 4 = ${2 ** 4}`);

// Comparison Operators: <, >, <=, >=, === (strict equality), !== (strict inequality)
let threshold = 0.75;
let isRelevant = score >= threshold; // >= (greater than or equal)
console.log(`Comparison: ${score} >= ${threshold} is ${isRelevant}`);

// === (Strict Equality): Checks for equality of value AND type. HIGHLY RECOMMENDED.
console.log(`Strict Equality (===): 5 === 5 is ${5 === 5}`);     // true (number === number, same value)
// @ts-expect-error TS2367: Comparison between types 'number' and 'string' which have no overlap. Intended for demonstration.
console.log(`Strict Equality (===): 5 === "5" is ${5 === "5"}`); // false (number !== string)

// !== (Strict Inequality): Checks for inequality of value OR type. HIGHLY RECOMMENDED.
console.log(`Strict Inequality (!==): 5 !== 5 is ${5 !== 5}`);     // false
// @ts-expect-error TS2367: Comparison between types 'number' and 'string' which have no overlap. Intended for demonstration.
console.log(`Strict Inequality (!==): 5 !== "5" is ${5 !== "5"}`); // true
// @ts-expect-error TS2367: Comparison between literal types '5' and '8' which have no overlap. Intended for demonstration.
console.log(`Strict Inequality (!==): 5 !== 8 is ${5 !== 8}`);     // true

// == (Loose Equality): Checks for equality after attempting type coercion. AVOID IF POSSIBLE - source of bugs.
// console.log(`Loose Equality (==): 5 == "5" is ${5 == "5"}`); // true (string "5" coerced to number 5) - AVOID!

// Logical Operators: && (AND), || (OR), ! (NOT)
// Similar to Python's `and`, `or`, `not`.
let hasSufficientDocs = numDocs >= 3;
let meetsThreshold = score > threshold;

let canGenerate = hasSufficientDocs && meetsThreshold; // Logical AND (&&)
console.log(`Logical AND: hasSufficientDocs && meetsThreshold is ${canGenerate}`);

let needsReview = !meetsThreshold || numDocs === 0; // Logical OR (||) and NOT (!)
console.log(`Logical OR/NOT: !meetsThreshold || numDocs === 0 is ${needsReview}`);


// ==========================================================================
// 5. RAG Context Examples
// ==========================================================================
console.log("\n--- 5. RAG Context Examples ---");

const ragSystemName: string = "DocuMind RAG";
let currentRagQuery: string | null = null; // Query might not be set yet
const ragMaxTokens: number = 4096;
const ragIsActive: boolean = true;

// Set the query
currentRagQuery = "What are the key challenges in implementing RAG systems?";

console.log(`System: ${ragSystemName}`);
console.log(`Active: ${ragIsActive}`);
console.log(`Max Tokens: ${ragMaxTokens}`);
console.log(`Current Query: ${currentRagQuery}`);

// --- End of Module 1 ---