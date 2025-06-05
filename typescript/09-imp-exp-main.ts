// 09-modules/main.ts

/**
 * Main execution file that imports and uses code from other modules.
 */

// Importing the Default Export: Give it any name you want (conventionally PascalCase for classes)
// Note the '.js' extension again, crucial for NodeNext module resolution.
import DocumentStore from './09-imp-exp-documentStore.js';
// You can also import named exports alongside the default
import { DEFAULT_STORE_NAME } from './09-imp-exp-documentStore.js';

// Importing Named Exports from types.ts: Use curly braces {}
import { BasicDocument, RetrievedChunk, ConfidenceScore } from './09-imp-exp-types.js';

// Importing Named Exports from utils.ts and renaming one with 'as'
import { calculateWordCount, formatContextForLLM as formatContext } from './09-imp-exp-utils.js';

// ALTERNATIVE: Importing everything from utils.ts into a single namespace object (less common)
// import * as StringUtils from './utils.js';
// Usage would then be: StringUtils.calculateWordCount(...)



console.log("--- Module 9: Modules (Import/Export) ---");
console.log(`Default store name constant: ${DEFAULT_STORE_NAME}`);

// --- Using the imported items ---

// Create an instance of the imported DocumentStore class
const store = new DocumentStore("MyProjectStore");

// Create documents using the imported BasicDocument interface/type
const doc1: BasicDocument = { id: "main-001", text: "This is the main document content." };
const doc2: BasicDocument = { id: "main-002", text: "Modules help organize code effectively." };

store.addDocument(doc1);
store.addDocument(doc2);

console.log("Document IDs in store:", store.listDocumentIds());
console.log(`Document count: ${store.documentCount}`);

// Use imported utility functions
const wordCount = calculateWordCount(doc1.text);
console.log(`Word count for doc1: ${wordCount}`);
// console.log(`Word count via namespace: ${StringUtils.calculateWordCount(doc2.text)}`); // If using namespace import

// Prepare some retrieved chunks using the imported type
const chunk1: RetrievedChunk = { ...doc1, chunkId: "c1", score: 0.95 }; // Use spread syntax
const chunk2: RetrievedChunk = { ...doc2, chunkId: "c1", score: 0.91 };

// Use the renamed imported function
const formatted = formatContext([chunk1, chunk2], 500);
console.log("\nFormatted context for LLM:");
console.log(formatted);

// --- End of Module 9 ---