// 11-generics/generics.ts

/**
 * Module 11: Generics
 * Covers: Writing reusable code components (functions, interfaces, classes)
 * that work with multiple types while maintaining type safety using type parameters (`<T>`).
 */

/**
 * What are Generics?
 * Generics allow you to write reusable code that can work with different types
 * without sacrificing type safety. They use type parameters (like `<T>` or `<Type>`)
 * as placeholders for specific types that will be provided when the code is used.
 * This avoids the need to write separate code for each type or resort to using `any`,
 * which loses type information.
 * 
 * Note: While 'T' is a common convention for "Type", you can use any valid identifier
 * inside the angle brackets `<...>`. Using descriptive names like `<ElementType>` or
 * `<ResponseType>` (as seen later in this file) can improve clarity.
 */

console.log("--- Module 11: Generics ---");



// ==========================================================================
// 1. The Problem: Reusability vs. Type Safety
// ==========================================================================
console.log("\n--- 1. The Problem: Reusability vs. Type Safety ---");

// Imagine a function that returns the input value (identity function).

// Approach 1: Specific types (Not reusable)
function identityNumber(arg: number): number { return arg; }
function identityString(arg: string): string { return arg; }
// We need a separate function for every type!

// Approach 2: Using `any` (Loses type safety!)
function identityAny(arg: any): any { return arg; }
let outputAny = identityAny("hello"); // outputAny is now 'any', we lose type info
// outputAny.thisMethodDoesNotExist(); // No compile error, potential runtime error!

// Approach 3: Using Generics (Reusable AND Type Safe!) - This is the solution!
// We introduce a 'type variable' (commonly `<T>`) that acts as a placeholder.

function identity<T>(arg: T): T {
  // `T` captures the type of the argument passed in.
  // The function accepts type `T` and returns the same type `T`.
  return arg;
}

// TypeScript infers the type `T` based on the argument:
let outputString = identity("hello TypeScript"); // TS infers T is string, outputString is string
let outputNumber = identity(123);              // TS infers T is number, outputNumber is number
let outputBoolean = identity(true);            // TS infers T is boolean, outputBoolean is boolean

console.log(`Generic identity (string): ${outputString.toUpperCase()}`); // OK, TS knows it's a string
console.log(`Generic identity (number): ${outputNumber.toFixed(2)}`);     // OK, TS knows it's a number
// outputBoolean.toUpperCase(); // Compile Error: Property 'toUpperCase' does not exist on type 'boolean'. (Type safety preserved!)



// ==========================================================================
// 2. Generic Functions
// ==========================================================================
console.log("\n--- 2. Generic Functions ---");

// Functions that use type variables `<T>` in their signature.

// Example: Get the first element of an array of any type.
function getFirstElement<ElementType>(arr: ElementType[]): ElementType | undefined {
  // Using a more descriptive name 'ElementType' instead of 'T' enhances readability here.
  return arr.length > 0 ? arr[0] : undefined;
}

const numbers = [10, 20, 30];
const firstNum = getFirstElement(numbers); // TS infers ElementType is number
console.log(`First number: ${firstNum}`); // Type is number | undefined

const strings = ["alpha", "beta", "gamma"];
const firstStr = getFirstElement(strings); // TS infers ElementType is string
console.log(`First string: ${firstStr}`);   // Type is string | undefined

const emptyArray: boolean[] = [];
const firstEmpty = getFirstElement(emptyArray); // TS infers ElementType is boolean
console.log(`First element of empty array: ${firstEmpty}`); // undefined



// ==========================================================================
// 3. Generic Interfaces and Type Aliases
// ==========================================================================
console.log("\n--- 3. Generic Interfaces and Type Aliases ---");

// We can define interfaces and type aliases that accept type parameters.

// Example: A standard wrapper for API responses
interface ApiResponse<DataType> { // DataType is the type parameter
  success: boolean;
  data: DataType | null; // The actual data payload depends on the specific API call
  error?: string;       // Optional error message
  timestamp: Date;
}

// Using the generic interface:
type User = { id: string; name: string; };
type Product = { sku: string; price: number; };

const userResponse: ApiResponse<User> = { // Specify User as the DataType
  success: true,
  data: { id: "user-1", name: "Alice" },
  timestamp: new Date()
};

const productResponse: ApiResponse<Product[]> = { // Specify Product[] as the DataType
  success: true,
  data: [{ sku: "TSHIRT-RED-L", price: 19.99 }, { sku: "MUG-CODE", price: 9.99 }],
  timestamp: new Date()
};

const errorResponse: ApiResponse<null> = { // Specify null if there's no data on error
  success: false,
  data: null,
  error: "Failed to connect to database",
  timestamp: new Date()
};

console.log("User API Response:", userResponse);
if (userResponse.success) {
  console.log(`  User Name: ${userResponse.data?.name}`); // Accessing data?.name is type-safe
}

console.log("Product List API Response:", productResponse);
console.log("Error API Response:", errorResponse);

// Generic Type Alias example
type DataPair<K, V> = { // Takes two type parameters
  key: K;
  value: V;
};

const pair1: DataPair<string, number> = { key: "age", value: 30 };
const pair2: DataPair<number, boolean> = { key: 123, value: true };
console.log("Generic Type Alias Pair:", pair1);



// ==========================================================================
// 4. Generic Classes
// ==========================================================================
console.log("\n--- 4. Generic Classes ---");

// Classes can also be generic, allowing them to work with different types.

class DataStore<T> {
  // This class stores an array of items of type T
  private data: T[] = [];

  add(item: T): void {
    this.data.push(item);
    console.log(`  Added item: ${item}. Store size: ${this.data.length}`);
  }

  getAll(): T[] {
    return [...this.data]; // Return a copy
  }

  find(predicate: (item: T) => boolean): T | undefined {
    // Accepts a function to find an item
    return this.data.find(predicate);
  }
}

// Creating instances with specific types:
const stringStore = new DataStore<string>(); // Store for strings
stringStore.add("TypeScript");
stringStore.add("Generics");
console.log("String Store Contents:", stringStore.getAll());

const numberStore = new DataStore<number>(); // Store for numbers
numberStore.add(100);
numberStore.add(200);
console.log("Number Store Contents:", numberStore.getAll());

const foundItem = stringStore.find(item => item.startsWith("Gen"));
console.log(`Found item starting with 'Gen': ${foundItem}`);



// ==========================================================================
// 5. Generic Constraints (`extends`)
// ==========================================================================
console.log("\n--- 5. Generic Constraints ---");

// Sometimes you need to guarantee that the generic type `T` has certain properties or methods.
// We use `extends` to constrain the type parameter.

// Example: A function that works on anything with a `length` property.
// We constrain T to be *at least* `{ length: number }`.
function logLength<T extends { length: number }>(item: T): void {
  console.log(`Item length: ${item.length}`);
}

logLength("Hello");    // Works (string has length)
logLength([1, 2, 3]);   // Works (array has length)
logLength({ length: 10, value: "abc" }); // Works (object has length property)
// logLength(123);       // Error: Argument of type 'number' is not assignable to parameter of type '{ length: number; }'.
// logLength({});        // Error: Argument of type '{}' is not assignable to parameter of type '{ length: number; }'.

// Another example: constrain to conform to an interface
interface Nameable {
  name: string;
}
function greet<T extends Nameable>(item: T): void {
  console.log(`Hello, ${item.name}!`);
}
greet({ name: "Alice", age: 30 }); // Works
// greet({ age: 30 }); // Error: Property 'name' is missing...



// ==========================================================================
// 6. RAG Context Examples
// ==========================================================================
console.log("\n--- 6. RAG Context Examples ---");

// Generic API Response type (as seen before)
interface RagApiResponse<T> {
  status: 'success' | 'error';
  data: T | null;
  errorMessage?: string;
}

// Type for retrieved documents (simplified)
type RetrievedDoc = { id: string; content: string; score: number };

// Simulate an API call function that uses the generic response type
async function fakeRagApiCall<ResponseType>(endpoint: string, payload: any): Promise<RagApiResponse<ResponseType>> {
  console.log(`  Calling fake RAG API endpoint: ${endpoint}...`);
  await new Promise(resolve => setTimeout(resolve, 300)); // Simulate delay
  // Simulate success/failure
  if (Math.random() > 0.1) { // 90% success
    // Simulate different response data based on endpoint
    let responseData: any;
    if (endpoint === '/retrieve') {
      responseData = [
        { id: 'doc-sim-1', content: '...', score: 0.9 },
        { id: 'doc-sim-2', content: '...', score: 0.8 }
      ];
    } else if (endpoint === '/status') {
      responseData = { system: 'RAG v2', health: 'OK' };
    } else {
      responseData = { message: 'Operation successful' };
    }
    return { status: 'success', data: responseData as ResponseType, errorMessage: undefined };
  } else {
    return { status: 'error', data: null, errorMessage: `Failed to call ${endpoint}` };
  }
}

// Using the generic function with specific types
async function runRagOperations() {
  console.log("Running RAG operations...");

  const retrievalResult = await fakeRagApiCall<RetrievedDoc[]>('/retrieve', { query: 'test' });
  if (retrievalResult.status === 'success' && retrievalResult.data) {
    console.log(`  Retrieved ${retrievalResult.data.length} documents.`);
    console.log(`  First doc ID: ${retrievalResult.data[0]?.id}`);
  } else {
    console.error(`  Retrieval failed: ${retrievalResult.errorMessage}`);
  }

  const statusResult = await fakeRagApiCall<{ system: string; health: string; }>('/status', {});
  if (statusResult.status === 'success' && statusResult.data) {
    console.log(`  System Status: ${statusResult.data.health} (System: ${statusResult.data.system})`);
  } else {
    console.error(`  Status check failed: ${statusResult.errorMessage}`);
  }
}

// Run the example
// (async () => {
//    await runRagOperations();
// })();


// --- End of Module 11 ---