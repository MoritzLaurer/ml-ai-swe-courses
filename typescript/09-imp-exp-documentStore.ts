// 09-modules/documentStore.ts

/**
 * Defines and exports the DocumentStore class.
 * It imports necessary types from 'types.ts'.
 */

// Importing Named Exports: Use curly braces {} and specify names.
// IMPORTANT: Notice the './types.js' path. Even though the source file is '.ts',
// when using Node.js native ES Modules (`"module": "NodeNext"` in tsconfig),
// TypeScript expects the import path to resolve to the *output* JavaScript file.
// The compiler requires you to write the '.js' extension in the import path.
import { BasicDocument, DocumentID } from './09-imp-exp-types.js';

// Using Default Export: Exporting a single 'main' item from this module.
// A file can have only ONE default export.
export default class DocumentStore {
  // Property visibility using 'private'
  private documents: Map<DocumentID, BasicDocument> = new Map(); // Use Map for efficient ID lookup
  public readonly storeName: string;

  constructor(name: string) {
    this.storeName = name;
    console.log(`(Module) DocumentStore "${this.storeName}" initialized.`);
  }

  addDocument(doc: BasicDocument): void {
    if (this.documents.has(doc.id)) {
      console.warn(`(Module) Document with ID "<span class="math-inline">\{doc\.id\}" already exists in "</span>{this.storeName}". Ignoring.`);
      return;
    }
    this.documents.set(doc.id, doc); // Use Map's set method
    console.log(`(Module) Document "<span class="math-inline">\{doc\.id\}" added to "</span>{this.storeName}". Total: ${this.documents.size}`);
  }

  getDocumentById(id: DocumentID): BasicDocument | undefined {
    const foundDoc = this.documents.get(id); // Use Map's get method
    if (!foundDoc) {
       console.log(`(Module) Document "<span class="math-inline">\{id\}" not found in "</span>{this.storeName}".`);
    }
    return foundDoc;
  }

  listDocumentIds(): DocumentID[] {
    // Get IDs from the Map keys iterator
    return Array.from(this.documents.keys());
  }

  get documentCount(): number {
    return this.documents.size;
  }
}

// Example of another named export from the same file (less common when there's a default class)
export const DEFAULT_STORE_NAME = "DefaultStore";