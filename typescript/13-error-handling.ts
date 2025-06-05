// 13-error-handling/error-handling.ts

/**
 * Module 13: Error Handling
 * Covers: Runtime errors, `throw`, `Error` object, `try...catch...finally` blocks,
 * handling unknown errors in catch, custom errors, async error handling.
 */

console.log("--- Module 13: Error Handling ---");

// TypeScript catches type errors at COMPILE time.
// Error handling deals with errors that happen at RUNTIME.



// ==========================================================================
// 1. Runtime Errors, `throw`, and the `Error` Object
// ==========================================================================
console.log("\n--- 1. Runtime Errors and `throw` ---");

// Errors can occur for many reasons: invalid operations, external systems failing, unexpected data.
// We can intentionally signal an error using the `throw` statement.

// It's BEST PRACTICE to throw instances of the built-in `Error` class or custom classes extending `Error`.
// An Error object typically contains a `message` and often a `stack` trace.

function checkAge(age: number): void {
  if (age < 0) {
    // Throwing an Error object
    throw new Error("Age cannot be negative.");
  }
  if (age < 18) {
    // Throwing a different type of error (less common for built-ins, more for custom)
    // throw new RangeError("User must be 18 or older."); // Another built-in Error type
    console.log("Age is valid but user is a minor.");
  } else {
    console.log(`Age ${age} is valid.`);
  }
}

// Calling this function directly without handling would crash the program if an error is thrown.
// checkAge(-5); // Uncommenting this would stop the script here if not caught.



// ==========================================================================
// 2. The `try...catch` Block
// ==========================================================================
console.log("\n--- 2. `try...catch` Block ---");

// To handle potential runtime errors gracefully, we use `try...catch`.

try {
  // The code that might throw an error goes inside the `try` block.
  console.log("Attempting to check age -5...");
  checkAge(-5); // This line will throw an error.
  console.log("This line will NOT be reached if an error occurs above.");
} catch (error) {
  // The `catch` block executes ONLY if an error is thrown in the `try` block.
  // The `error` variable contains the value that was thrown.

  // IMPORTANT: By default in modern TypeScript (with strict settings),
  // the type of `error` in a catch block is `unknown`.
  // This is because technically *anything* can be thrown (not just Error objects).
  // We need to safely check its type before using its properties.
  console.error("--- ERROR Caught! ---");

  if (error instanceof Error) {
    // Safely check if it's an instance of the Error class
    console.error(`Caught an Error object: ${error.message}`);
    // console.error(error.stack); // Can also log stack trace if needed
  } else if (typeof error === 'string') {
    console.error(`Caught a string error: ${error}`);
  } else {
    console.error(`Caught an unknown type of error:`, error);
  }
}

console.log("Program continues after handling the error.");



// ==========================================================================
// 3. The `finally` Block
// ==========================================================================
console.log("\n--- 3. `finally` Block ---");

// The `finally` block executes AFTER the `try` block (and `catch` if an error occurred),
// regardless of whether an error was thrown or caught.
// Useful for cleanup code (e.g., closing files, releasing resources, logging completion).

let resourceAcquired = false;
try {
  console.log("Attempting an operation that might succeed or fail...");
  resourceAcquired = true; // Simulate acquiring a resource
  console.log("Resource acquired.");

  // Simulate potential failure
  if (Math.random() < 0.5) {
    throw new Error("Simulated failure during operation.");
  }

  console.log("Operation completed successfully.");

} catch (error) {
  console.error("--- ERROR Caught during operation ---");
  if (error instanceof Error) {
    console.error(`  Message: ${error.message}`);
  } else {
    console.error(`  Unknown error:`, error);
  }
} finally {
  // This code ALWAYS runs.
  console.log("--- finally Block Executing ---");
  if (resourceAcquired) {
    console.log("  Releasing resource...");
    resourceAcquired = false; // Cleanup
  } else {
    console.log("  No resource was acquired.");
  }
  console.log("--- Cleanup finished ---");
}



// ==========================================================================
// 4. Custom Error Classes
// ==========================================================================
console.log("\n--- 4. Custom Error Classes ---");

// For more specific error handling, you can create custom error classes by extending `Error`.

class ValidationError extends Error {
  public field?: string; // Add custom properties

  constructor(message: string, field?: string) {
    super(message); // Call the base Error constructor
    this.name = "ValidationError"; // Set the error name
    this.field = field;

    // Fix for prototype chain in older environments (often needed for custom errors)
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

class ApiError extends Error {
  public statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

function parseUserInput(input: any): void {
  if (typeof input.name !== 'string' || input.name.length === 0) {
    throw new ValidationError("Name is required and must be a string.", "name");
  }
  if (typeof input.age !== 'number' || input.age <= 0) {
    throw new ValidationError("Age must be a positive number.", "age");
  }
  console.log(`User input parsed successfully:`, input);
}

try {
  // parseUserInput({ name: "Alice", age: 30 }); // This would succeed
  parseUserInput({ name: "Bob" }); // This will throw ValidationError
} catch (error) {
  console.error("--- ERROR Caught during input parsing ---");
  if (error instanceof ValidationError) {
    console.error(`Validation Error: ${error.message} (Field: ${error.field ?? 'N/A'})`);
  } else if (error instanceof Error) {
    console.error(`Generic Error: ${error.message}`);
  } else {
    console.error(`Unknown error:`, error);
  }
}



// ==========================================================================
// 5. Error Handling in Async/Await Code (Recap)
// ==========================================================================
console.log("\n--- 5. Error Handling in Async/Await ---");

// As seen in Module 10, `try...catch` works seamlessly with `await`.
// If the awaited Promise rejects, `await` throws that rejection, which `catch` can handle.

async function simulateFailableAsync(shouldFail: boolean): Promise<string> {
  await new Promise(resolve => setTimeout(resolve, 100)); // Simulate delay
  if (shouldFail) {
    throw new ApiError("Simulated API failure in async function", 503);
  }
  return "Async operation succeeded!";
}

async function runAsyncWithErrorHandling() {
  console.log("Running async function that might fail...");
  try {
    const result = await simulateFailableAsync(true); // Set to true to trigger error
    console.log(`Success: ${result}`); // This won't be reached if it fails
  } catch (error) {
    console.error("--- ASYNC ERROR Caught ---");
    if (error instanceof ApiError) {
      console.error(`API Error: ${error.message} (Status Code: ${error.statusCode})`);
    } else if (error instanceof Error) {
      console.error(`Generic Error: ${error.message}`);
    } else {
      console.error(`Unknown async error:`, error);
    }
  } finally {
    console.log("--- Async operation attempt finished ---");
  }
}

// Need to await the async function or use .then() if called from non-async context
// Using an IIFE again for top-level await simulation
// (async () => {
//   await runAsyncWithErrorHandling();
// })();



// ==========================================================================
// 6. RAG Context Examples
// ==========================================================================
console.log("\n--- 6. RAG Context Examples ---");

// Assume we have a function to retrieve document content, which might fail

// Custom error for retrieval issues
class RetrievalError extends Error {
  constructor(message: string, public docId?: string) {
    super(message);
    this.name = "RetrievalError";
    Object.setPrototypeOf(this, RetrievalError.prototype);
  }
}

function getDocumentContent(docId: string): string {
  console.log(`Attempting to retrieve content for doc: ${docId}`);
  // Simulate failure for certain IDs
  if (docId === "doc-not-found" || docId === "") {
    throw new RetrievalError("Document could not be found or ID is invalid.", docId);
  }
  if (docId === "doc-access-denied") {
      throw new Error("Access denied for this document."); // Throwing a generic error
  }
  // Simulate success
  return `Content of document ${docId}. Lorem ipsum...`;
}

function processSingleDoc(docId: string): void {
  try {
    const content = getDocumentContent(docId);
    console.log(`Successfully retrieved content for ${docId}: "${content.substring(0, 30)}..."`);
    // ... further processing ...
  } catch (error) {
    console.error(`--- Failed to process document '${docId}' ---`);
    if (error instanceof RetrievalError) {
      console.error(`  Retrieval Error: ${error.message}`);
    } else if (error instanceof Error) {
       console.error(`  Generic Error: ${error.message}`);
    } else {
      console.error("  Unknown error occurred.");
    }
    // Potentially log this error, mark doc as failed, etc.
  }
}

processSingleDoc("doc-valid-123");
processSingleDoc("doc-not-found");
processSingleDoc("doc-access-denied");
processSingleDoc("");


// --- End of Module 13 ---