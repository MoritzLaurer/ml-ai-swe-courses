// 03-objects/objects.ts

/**
 * Module 3: JavaScript Objects
 * Covers: Creating object literals, accessing and modifying properties, nested objects.
 */

console.log("--- Module 3: JavaScript Objects ---");



// ==========================================================================
// 1. Creating Object Literals
// ==========================================================================
console.log("\n--- 1. Creating Object Literals ---");

// Objects are collections of key-value pairs, enclosed in curly braces `{}`.
// Keys are typically strings (quoted or unquoted if valid identifiers) or Symbols.
// Values can be any type: primitives, arrays, functions, other objects.
// This is the primary way to group related data and functionality.

const documentV1 = {
  id: "doc-v1-001", // key: value
  title: "Introduction to RAG",
  year: 2024,
  isPublished: true,
  keywords: ["AI", "NLP", "RAG"], // Value can be an array
};

// TypeScript *infers* a type shape for this object based on its properties and their initial values.
// Hovering over `documentV1` in VS Code would show something like:
// const documentV1: { id: string; title: string; year: number; isPublished: boolean; keywords: string[]; }

console.log("Simple Object Literal:", documentV1);



// ==========================================================================
// 2. Accessing Properties
// ==========================================================================
console.log("\n--- 2. Accessing Properties ---");

// You can access the values (properties) of an object using:
// a) Dot Notation (`object.propertyName`) - Preferred when the key is a valid identifier.
console.log(`Using Dot Notation - Title: ${documentV1.title}`);
console.log(`Using Dot Notation - Year: ${documentV1.year}`);

// b) Bracket Notation (`object['propertyName']`) - Required when the key is not a valid identifier
//    (e.g., contains spaces, starts with a number) or when the key is stored in a variable.
console.log(`Using Bracket Notation - ID: ${documentV1['id']}`);

// Example with a variable key:
const keyToAccess = 'keywords';
console.log(`Accessing '${keyToAccess}' via Bracket Notation: ${documentV1[keyToAccess]}`); // Accesses the keywords array

// Accessing a non-existent property returns `undefined`
// console.log(`Accessing non-existent property: ${documentV1.author}`); // In TS, this would likely cause a compile error if type checking is strict, because the inferred type doesn't have 'author'. In plain JS, it would be `undefined`.



// ==========================================================================
// 3. Modifying Properties and Adding New Ones
// ==========================================================================
console.log("\n--- 3. Modifying Properties and Adding New Ones ---");

// If the object variable was declared with `let` (or `const` but modifying internal properties), you can change values.
documentV1.isPublished = false; // Modify existing property
documentV1.keywords.push("Search"); // Modify the array *inside* the object

console.log(`Modified isPublished: ${documentV1.isPublished}`);
console.log(`Modified keywords: ${documentV1.keywords}`);

// Adding new properties:
// In plain JavaScript, you can add properties freely.
// In TypeScript, if you haven't defined the object's shape to allow extra properties
// (e.g., using index signatures, which we'll see later, or using `any`),
// adding arbitrary properties will often cause a compile-time error.

// Let's try adding a property (this might error depending on TS strictness / inference)
// Since TS inferred a specific shape for documentV1 initially, adding a new property might be flagged.
// To allow this behaviour more explicitly, we often define types/interfaces.
// (documentV1 as any).source = "Blog Post"; // Using 'as any' bypasses type checking (AVOID!)
// console.log("Object after adding 'source' (using 'as any'):", documentV1);



// ==========================================================================
// 4. Nested Objects
// ==========================================================================
console.log("\n--- 4. Nested Objects ---");

// Object properties can themselves be other objects.

const retrievalMetadata = {
  retrievedAt: new Date(),
  sourceSystem: { // Nested object
    name: "Internal Search Index",
    version: "2.1"
  },
  confidence: 0.85
};

console.log("Nested Object:", retrievalMetadata);

// Accessing properties in nested objects:
console.log(`Retrieval Confidence: ${retrievalMetadata.confidence}`);
console.log(`Source System Name: ${retrievalMetadata.sourceSystem.name}`); // Chain dot notation
console.log(`Source System Version (via bracket): ${retrievalMetadata['sourceSystem']['version']}`);



// ==========================================================================
// 5. Objects with Functions (Methods) - Brief Preview
// ==========================================================================
console.log("\n--- 5. Objects with Functions (Methods) ---");

// Object properties can also be functions. When a function is part of an object, it's called a method.
const ragSystem = {
  name: "DocuMind RAG",
  status: "idle",
  start: function() { // A function assigned to the 'start' key
    this.status = "running"; // 'this' refers to the object itself (ragSystem)
    console.log(`${this.name} system starting... Status: ${this.status}`);
  },
  // Shorthand method syntax (more common now):
  stop() {
    this.status = "stopped";
    console.log(`${this.name} system stopping... Status: ${this.status}`);
  }
};

ragSystem.start(); // Call the method using ()
console.log(`Current status after start: ${ragSystem.status}`);
ragSystem.stop();
console.log(`Current status after stop: ${ragSystem.status}`);

// We will explore methods more when we discuss Classes and Functions in detail.

// --- End of Module 3 ---