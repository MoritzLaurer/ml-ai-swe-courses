// 06-functions/functions.ts

/**
 * Module 6: Functions
 * Covers: Defining functions (declarations, expressions, arrows), parameter & return type
 * annotations, optional/default/rest parameters, and function types.
 */

// Import types from previous modules if needed (adjust path if necessary)
// We'll use a simple type alias here for demonstration if we don't import
type RetrievedChunk = { readonly documentId: string; readonly chunkId: string; text: string; score: number; };

console.log("--- Module 6: Functions ---");



// ==========================================================================
// 1. Defining Functions: Syntax Options
// ==========================================================================
console.log("\n--- 1. Defining Functions: Syntax Options ---");

// Option A: Function Declaration
// - Uses the `function` keyword.
// - Hoisted: Can be called before it's defined in the code (though generally good practice to define first).
// - Requires explicit return type annotation (or TS infers, but explicit is clearer).
function calculateWordCount(text: string): number {
  if (!text) {
    return 0;
  }
  return text.trim().split(/\s+/).length;
}
console.log(`Function Declaration: Word count = ${calculateWordCount(" Hello TypeScript world! ")}`);

// Option B: Function Expression
// - Assigns an anonymous function to a variable (`const` or `let`).
// - Not Hoisted: Cannot be called before the assignment.
// - Often used when functions are treated as values (passed around, assigned conditionally).
const formatSummary = function(title: string, score: number): string {
  return `Summary for "${title}": Score = ${(score * 100).toFixed(1)}%`;
};
console.log(`Function Expression: ${formatSummary("RAG Basics", 0.876)}`);

// Option C: Arrow Function Expression (ES6+)
// - Concise syntax (`=>`).
// - Not Hoisted.
// - Lexical `this` binding: Arrow functions don't have their own `this`. They capture the
//   `this` value of the enclosing lexical context (the code surrounding where the arrow
//   function is *defined*).
//   Advantage in Classes/Callbacks: Unlike regular functions, whose `this` depends
//   on *how they are called*, an arrow function's `this` remains fixed to its
//   definition context. This is useful for methods (defined as class properties)
//   that are passed as callbacks (e.g., event handlers, setTimeout) because it avoids
//   losing the intended `this` (the class instance) and eliminates the need for `.bind(this)`.
// - Brevity: No `function` keyword, optional parentheses for single parameter, implicit return for single expression.
// - Often the preferred syntax for function expressions due to brevity and `this` behavior.
const generateQueryId = (base: string): string => {
  const timestamp = Date.now();
  return `${base}-${timestamp}`;
};
console.log(`Arrow Function: Query ID = ${generateQueryId("userQuery")}`);

// Arrow function with single expression (implicit return if no curly braces):
const isScoreAcceptable = (score: number, threshold: number): boolean => score >= threshold;
console.log(`Arrow Function (implicit return): Is 0.8 acceptable (threshold 0.75)? ${isScoreAcceptable(0.8, 0.75)}`);



// ==========================================================================
// 2. Return Type Annotations (and `void`)
// ==========================================================================
console.log("\n--- 2. Return Type Annotations ---");

// Explicitly annotating the return type (`: type` after parentheses `()`) is crucial for clarity and safety.

// TypeScript *can* infer return types often, but being explicit makes the function's contract clear.
function addScores(s1: number, s2: number) { // Return type `number` is inferred here
  return s1 + s2;
}
const totalScore = addScores(0.5, 0.3); // TS knows totalScore is number

// Use `: void` for functions that do not return a value.
function logMessage(message: string, level: string = "info"): void {
  console.log(`[${level.toUpperCase()}] ${message}`);
  // No `return` statement, or `return;` without a value is allowed.
  // return undefined; // Also valid for void
}
logMessage("Function module started.");



// ==========================================================================
// 3. Optional, Default, and Rest Parameters
// ==========================================================================
console.log("\n--- 3. Optional, Default, and Rest Parameters ---");

// Optional Parameters (`?`):
// - Must come *after* all required parameters.
// - Inside the function, the parameter's value might be `undefined`.
function greetUser(name: string, greeting?: string): void {
  const finalGreeting = greeting ? greeting : "Hello"; // Use default if greeting is undefined
  console.log(`${finalGreeting}, ${name}!`);
}
greetUser("Alice"); // Output: Hello, Alice!
greetUser("Bob", "Good morning"); // Output: Good morning, Bob!

// Default Parameters (`= value`):
// - Makes the parameter optional. If not provided, the default value is used.
// - Type is often inferred from the default value.
// - Can appear anywhere in the parameter list (unlike optional `?`).
function createRAGRequest(query: string, topK: number = 3, source: string = "default_index"): object {
  console.log(`Creating request: query="${query}", topK=${topK}, source=${source}`);
  return { query, topK, source }; // Returns an object (type inferred as object)
}
createRAGRequest("What is TS?"); // Uses defaults for topK and source
createRAGRequest("Explain JS Objects", 5); // Overrides topK, uses default source
createRAGRequest("Compare Python and JS", 2, "web_search"); // Overrides both defaults

// Rest Parameters (`...name: type[]`):
// - Collects multiple arguments into a single array.
// - Must be the *last* parameter in the function signature.
// - Provides convenience for the caller, allowing them to pass multiple arguments directly
//   instead of manually creating an array.
//   e.g., `combineTexts(" | ", "a", "b")` instead of `combineTexts(" | ", ["a", "b"])` if using a standard array param.
function combineTexts(separator: string, ...texts: string[]): string {
  console.log(`Combining texts:`, texts);
  return texts.join(separator);
}
const combined = combineTexts(" | ", "First part.", "Second part.", "Third part.");
console.log(`Combined Result: ${combined}`);
const singleText = combineTexts(" | ", "Only one part."); // texts array will contain ["Only one part."]
console.log(`Combined Result (single): ${singleText}`);



// ==========================================================================
// 4. Function Types
// ==========================================================================
console.log("\n--- 4. Function Types ---");

// We can define the "shape" or "signature" of a function using a type alias.
// This is useful for assigning functions to variables or passing them as parameters (callbacks).

// Syntax: type FuncTypeName = (param1: type, param2: type) => returnType;

type StringFormatter = (input: string) => string;
type ScoreCalculator = (scores: number[]) => number;

// Using the function types:
const toUpperCaseFormatter: StringFormatter = (text) => {
  return text.toUpperCase();
};

const calculateAverage: ScoreCalculator = (scores) => {
  if (scores.length === 0) return 0;
  const sum = scores.reduce((acc, current) => acc + current, 0); // .reduce is an array method
  return sum / scores.length;
};

console.log(`Formatter applied: ${toUpperCaseFormatter("hello world")}`);
console.log(`Average score calculated: ${calculateAverage([0.8, 0.9, 0.7])}`);

// Using function types for parameters (callbacks):
function processScores(scores: number[], calculator: ScoreCalculator): void {
  const result = calculator(scores);
  logMessage(`Calculation result: ${result}`);
}
processScores([0.9, 0.95, 0.88], calculateAverage); // Pass calculateAverage as the callback



// ==========================================================================
// 5. RAG Context Examples
// ==========================================================================
console.log("\n--- 5. RAG Context Examples ---");

// Function to format context chunks into a single string for the LLM
function formatContextForLLM(chunks: RetrievedChunk[], maxChars?: number): string {
  let combinedText = "";
  for (const chunk of chunks) {
    const chunkText = `Document ID: ${chunk.documentId}\nScore: ${chunk.score.toFixed(2)}\nContent: ${chunk.text}\n\n`;
    if (maxChars !== undefined && (combinedText.length + chunkText.length > maxChars)) {
      logMessage("Reached character limit, stopping context formatting.", "warn");
      break; // Stop adding chunks if limit exceeded
    }
    combinedText += chunkText;
  }
  return combinedText.trim();
}

// Example usage:
const sampleChunks: RetrievedChunk[] = [
  { documentId: "rag-intro", chunkId: "c1", text: "RAG combines retrieval...", score: 0.92 },
  { documentId: "ts-basics", chunkId: "c1", text: "TypeScript adds types...", score: 0.88 }
];

const formattedContext = formatContextForLLM(sampleChunks);
console.log("Formatted Context:\n", formattedContext);

const limitedContext = formatContextForLLM(sampleChunks, 100); // With character limit
console.log("\nLimited Formatted Context:\n", limitedContext);

// Function type for a function that decides if retrieval is needed
type RetrievalDecisionMaker = (query: string, historyLength: number) => boolean;

const simpleDecisionMaker: RetrievalDecisionMaker = (query, historyLength) => {
  // Simple logic: retrieve if query is long enough or history is short
  return query.length > 10 || historyLength < 3;
};

console.log(`Need retrieval for 'Short query', history=1? ${simpleDecisionMaker('Short query', 1)}`); // true (history < 3)
console.log(`Need retrieval for 'Longer example query text', history=5? ${simpleDecisionMaker('Longer example query text', 5)}`); // true (query.length > 10)

// --- End of Module 6 ---