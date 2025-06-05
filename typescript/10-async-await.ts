// 10-async/async-await.ts

/**
 * Module 10: Handling Asynchronous Operations - Promises and Async/Await
 * Covers: Asynchronous concept, Promises (.then, .catch, .finally),
 * async functions, await keyword, error handling with try/catch, Promise.all.
 */

console.log("--- Module 10: Asynchronous Operations ---");



// ==========================================================================
// 1. The Asynchronous Concept & Simulating Async Work
// ==========================================================================
console.log("\n--- 1. Simulating Asynchronous Work ---");

// Synchronous code executes line by line, blocking further execution until it finishes.
// Asynchronous code starts an operation (e.g., network request, timer) and lets
// other code run while it waits. A callback or Promise handles the result later.

// We use `setTimeout` to simulate a task that takes time (like an API call).
// `setTimeout` itself is asynchronous.

function simulateApiCall(query: string, delayMs: number): Promise<string> {
  console.log(`  Starting API call for query: "${query}" (will take ${delayMs}ms)`);

  // A Promise represents the *eventual* result of an async operation.
  // It starts in a 'pending' state.
  // It can later transition to 'fulfilled' (resolved with a value) or 'rejected' (failed with an error).
  return new Promise((resolve, reject) => {
    // We simulate success/failure randomly for demonstration
    const success = Math.random() > 0.2; // 80% chance of success

    setTimeout(() => {
      if (success) {
        const response = `Data for "${query}" fetched successfully!`;
        console.log(`  API call for "${query}" succeeded.`);
        resolve(response); // Fulfill the promise with the response value
      } else {
        const errorMsg = `Failed to fetch data for "${query}"!`;
        console.error(`  API call for "${query}" failed.`);
        reject(new Error(errorMsg)); // Reject the promise with an Error object
      }
    }, delayMs);
  });
}



// ==========================================================================
// 2. Working with Promises: .then(), .catch(), .finally()
// ==========================================================================
console.log("\n--- 2. Using Promises (.then / .catch) ---");

// NOTE: This section demonstrates the "older" way of handling Promises using
// chained .then(), .catch(), and .finally() methods. While functional,
// this approach can lead to less readable code ("callback hell") for complex
// asynchronous flows and less intuitive error handling compared to async/await.
// It's included here primarily for understanding the history and mechanics of Promises.

console.log("Making API call (Promise chain method)...");
simulateApiCall("users", 1000)
  .then((responseData) => {
    // This function runs ONLY if the promise is fulfilled (resolved).
    console.log("  .then() callback executed:");
    console.log(`    Response Data: ${responseData}`);
    // You can chain .then() - the return value of one .then becomes the input for the next
    return responseData.toUpperCase();
  })
  .then((uppercasedData) => {
    console.log(`    Uppercased Data: ${uppercasedData}`);
  })
  .catch((error) => {
    // This function runs ONLY if the promise is rejected.
    console.error("  .catch() callback executed:");
    // It's good practice to check if the error is an instance of Error
    if (error instanceof Error) {
      console.error(`    Error Message: ${error.message}`);
    } else {
      console.error(`    Caught unexpected error: ${error}`);
    }
  })
  .finally(() => {
    // This function runs *always*, whether the promise fulfilled or rejected.
    // Useful for cleanup tasks (e.g., closing connections, hiding loaders).
    console.log("  .finally() callback executed. API call attempt finished.");
  });

console.log("... Code execution continues while API call is pending ...");



// ==========================================================================
// 3. `async` Functions
// ==========================================================================
console.log("\n--- 3. `async` Functions ---");

// NOTE: Starting from here, we introduce `async` and `await`, which represent
// the MODERN and RECOMMENDED approach for handling asynchronous operations
// in TypeScript and JavaScript.

// The `async` keyword before a function declaration makes it automatically return a Promise.
// The resolved value of the promise is whatever the function returns.
// If the function throws an error, the promise is rejected with that error.

async function simpleAsyncFunction(succeed: boolean): Promise<string> {
  console.log("  Inside simpleAsyncFunction...");
  if (succeed) {
    return "Operation successful!"; // This string becomes the resolved value of the promise
  } else {
    throw new Error("Operation failed!"); // Throwing an error rejects the promise
  }
}

// Calling it returns a promise
simpleAsyncFunction(true)
  .then(result => console.log(`  Async function success: ${result}`))
  .catch(err => console.error(`  Async function error: ${err.message}`));

simpleAsyncFunction(false)
  .then(result => console.log(`  Async function success: ${result}`)) // This won't run
  .catch(err => console.error(`  Async function error: ${err.message}`)); // This will run



// ==========================================================================
// 4. `await` Keyword
// ==========================================================================
console.log("\n--- 4. `await` Keyword ---");

// NOTE: `await` is used within `async` functions to pause execution until a
// Promise settles. This makes asynchronous code look and behave more like
// synchronous code, significantly improving READABILITY.

// The `await` keyword can ONLY be used inside an `async` function.
// It pauses the execution of the `async` function until the awaited Promise settles.
// - If the Promise fulfills, `await` returns the resolved value.
// - If the Promise rejects, `await` throws the rejection error (which can be caught).
// This makes asynchronous code look and behave a bit more like synchronous code.

async function processUserData() {
  console.log("  Processing user data using async/await...");
  console.log("  Fetching user profile...");
  // Instead of .then(), we use await:
  const userProfile = await simulateApiCall("userProfile", 500); // Pause here until promise resolves
  console.log(`    Profile fetched: "${userProfile}"`); // Execution resumes here

  console.log("  Fetching user orders...");
  const userOrders = await simulateApiCall("userOrders", 800); // Pause again
  console.log(`    Orders fetched: "${userOrders}"`);

  console.log("  User data processing complete.");
  return { profile: userProfile, orders: userOrders }; // Return value wraps in a Promise
}

// Need to call the async function and handle its resulting promise
// processUserData()
//   .then(data => console.log("  Final processed data:", data))
//   .catch(error => console.error("  Error processing user data:", error.message));



// ==========================================================================
// 5. Error Handling with `try...catch` and `async/await`
// ==========================================================================
console.log("\n--- 5. Error Handling with try...catch ---");

// NOTE: Using standard `try...catch...finally` blocks with `async/await` is the
// RECOMMENDED way to handle errors in asynchronous code. It's generally more
// intuitive and consistent with synchronous error handling compared to `.catch()` chains.

// Because `await` throws an error when a promise rejects, we can use
// standard `try...catch` blocks for error handling, which is often cleaner than `.catch()`.

async function processUserDataWithTryCatch() {
  console.log("  Processing user data (with try...catch)...");
  try {
    console.log("  Fetching user profile...");
    const userProfile = await simulateApiCall("userProfile", 600); // Try to await
    console.log(`    Profile fetched: "${userProfile}"`);

    console.log("  Fetching user orders (might fail)...");
    const userOrders = await simulateApiCall("userOrders", 900); // Try to await (this call might reject)
    console.log(`    Orders fetched: "${userOrders}"`);

    console.log("  User data processing complete.");
    return { profile: userProfile, orders: userOrders };

  } catch (error) {
    console.error("  --- ERROR Caught in processUserDataWithTryCatch ---");
    if (error instanceof Error) {
      console.error(`    Error message: ${error.message}`);
    } else {
      console.error(`    Caught unexpected error: ${error}`);
    }
    // Can decide to return a default value, re-throw, or just log
    return null; // Indicate failure by returning null
  } finally {
    console.log("  --- User data processing attempt finished (finally block) ---");
  }
}

// Call the function with try/catch
// processUserDataWithTryCatch()
//   .then(data => {
//     if (data) {
//       console.log("  Final processed data (try/catch):", data);
//     } else {
//       console.log("  Processing failed, returned null.");
//     }
//   });



// ==========================================================================
// 6. Concurrent Operations with `Promise.all()`
// ==========================================================================
console.log("\n--- 6. Concurrent Operations with Promise.all ---");

// NOTE: `Promise.all()` is essential for running multiple asynchronous operations
// concurrently. It's often used with `await` within an `async` function for
// waiting on multiple parallel tasks, which is a common and recommended pattern.

// Sometimes you want to run multiple async operations concurrently and wait for all to finish.
// `Promise.all()` takes an array of Promises and returns a new Promise that:
// - Fulfills when ALL input promises fulfill, returning an array of their results in order.
// - Rejects immediately if ANY of the input promises reject.

async function fetchMultipleResources() {
  console.log("  Fetching multiple resources concurrently...");
  try {
    const results = await Promise.all([
      simulateApiCall("resource A", 700),
      simulateApiCall("resource B", 1200), // This one takes longer
      simulateApiCall("resource C", 500)
    ]);
    // This line is reached only if ALL promises resolve successfully
    console.log("  All resources fetched successfully!");
    results.forEach((res, index) => console.log(`    Result ${index}: ${res}`));
    return results;
  } catch (error) {
    console.error("  --- ERROR Caught during Promise.all ---");
    if (error instanceof Error) {
      console.error(`    Error message: ${error.message}`); // Will show the error from the *first* promise that rejected
    }
    return null;
  }
}

// Call the concurrent function
// fetchMultipleResources(); // Run this to see concurrent calls



// ==========================================================================
// 7. RAG Context Examples
// ==========================================================================
console.log("\n--- 7. RAG Context Examples ---");

// NOTE: This practical example demonstrates how `async/await` and `try...catch`
// create clean, readable, and robust asynchronous logic for a more complex task
// like a RAG pipeline. This structure is highly recommended for real-world applications.

// Simulate async retrieval of documents
async function retrieveDocuments(query: string): Promise<string[]> {
  console.log(`  Retrieving documents for query: "${query}"...`);
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 800));

  // Simulate finding some document IDs
  const ids = [`doc_${query}_1`, `doc_${query}_2`, `doc_${query}_3`];
  console.log(`  Found document IDs: ${ids.join(', ')}`);
  return ids;
}

// Simulate async call to an LLM for generation
async function generateResponse(context: string, query: string): Promise<string> {
  console.log(`  Generating response for query: "${query}" using provided context...`);
  // Simulate network delay for LLM call
  await new Promise(resolve => setTimeout(resolve, 1500));

  const response = `LLM Response based on query "${query}" and context starting with: "${context.substring(0, 50)}..."`;
  console.log(`  LLM generation complete.`);
  return response;
}

// Main RAG processing function using async/await
async function handleQuery(query: string): Promise<void> {
  console.log(`--- Handling RAG Query: "${query}" ---`);
  try {
    const documentIds = await retrieveDocuments(query);
    // In a real app, you'd fetch content for these IDs, format context etc.
    const simplifiedContext = `Context based on documents: ${documentIds.join(', ')}`;

    const finalResponse = await generateResponse(simplifiedContext, query);

    console.log(`--- Final RAG Response ---`);
    console.log(finalResponse);

  } catch (error) {
    console.error("--- ERROR during RAG processing ---");
    if (error instanceof Error) {
      console.error(`  Message: ${error.message}`);
    }
  } finally {
      console.log(`--- RAG Query handling finished for: "${query}" ---`);
  }
}

// Execute the main RAG handler function
// We need to wrap the top-level call slightly differently now because Node doesn't allow top-level await by default without specific flags or in REPL
// Simple approach: create an async IIFE (Immediately Invoked Function Expression)
(async () => {
  console.log("\n--- Starting Async Function Calls (might interleave) ---");
  // Call functions that use .then/.catch
  simulateApiCall("test-1", 300)
    .then(r => console.log(`Promise chain result: ${r}`))
    .catch(e => console.error(`Promise chain error: ${e.message}`))
    .finally(() => console.log("Promise chain finished."));

  // Call functions that use async/await internally but need handling
  await processUserDataWithTryCatch(); // Wait for this one to finish
  await fetchMultipleResources(); // Wait for this one
  await handleQuery("What is async/await?"); // Wait for the full RAG example

  console.log("\n--- All Async operations initiated in IIFE potentially finished ---");
})();


// --- End of Module 10 ---