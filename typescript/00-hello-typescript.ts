// 00-setup/hello-typescript.ts

/**
 * This is a simple TypeScript script for Module 0.
 * It demonstrates basic syntax and how comments work.
 */

// Define a variable `message` and explicitly annotate its type as `string`.
// While TypeScript can often *infer* types, explicit annotation is good practice for clarity, especially for function signatures.
let message: string = "Hello, TypeScript World!";

// Define a function that takes a string parameter and prints it.
// We annotate the parameter `msg` as type `string`.
// We also annotate the function's return type as `void`, meaning it doesn't return any value.
function logMessage(msg: string): void {
  console.log(`Logging message: ${msg}`);
  // console.log is the standard way to print output to the console in Node.js/JavaScript, similar to Python's print().
  // The backticks (`) denote a template literal, allowing embedded expressions like ${msg}.
}

// Call the function with our message variable.
logMessage(message);

// Let's try assigning a non-string value to `message`
// message = 123; // Uncomment this line!

/*
If you uncomment the line above and try to compile, TypeScript will give you an error:
"Type 'number' is not assignable to type 'string'."
This is the core benefit of TypeScript - catching type errors *before* you run the code!
Comment it back out to proceed with successful compilation.
*/

console.log("TypeScript setup seems to be working!");