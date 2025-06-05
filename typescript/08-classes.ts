// 08-classes/classes.ts

/**
 * Module 8: Building Blueprints - Classes & Basic OOP
 * Covers: Defining classes, constructors, properties (fields), methods,
 * `this` keyword, instantiation (`new`), access modifiers (public, private, protected).
 */

// Let's use a simple interface from previous concepts for context
interface BasicDocument {
  id: string;
  text: string;
}
  
console.log("--- Module 8: Classes & Basic OOP ---");

// Classes are blueprints for creating objects with predefined properties and methods.
// They are central to Object-Oriented Programming (OOP).



// ==========================================================================
// 1. Basic Class Definition, Constructor, Properties, Methods
// ==========================================================================
console.log("\n--- 1. Basic Class Definition ---");

// ! section 5 shows a more concise approach to defining properties and the constructor of a class (specific to TS)

class DocumentStore {
    // --- Properties (Fields) ---
    // Variables that hold the state of an object created from the class.
    // Need type annotations. Can be initialized here or in the constructor.
    storeName: string = "Default Document Store"; // Initialized directly
    documents: BasicDocument[] = []; // Initialized with an empty array
  
    // --- Constructor ---
    // A special method called when creating a new instance (`new ClassName()`).
    // Used to initialize object properties. Parameters can be passed during creation.
    // similar to init in Python
    constructor(name: string) {
      this.storeName = name; // `this` refers to the specific instance being created.
      console.log(`DocumentStore "${this.storeName}" initialized.`);
      // Properties not initialized above (like `documents`) should ideally be initialized here if not done at declaration.
      // this.documents = []; // Already done above, but could be done here too.
    }
  
    // --- Methods ---
    // Functions defined within the class that define its behavior.
    // They can access and modify the object's properties using `this`.
    addDocument(doc: BasicDocument): void {
      // Check if document with same ID already exists (using an array method)
      if (this.documents.some(d => d.id === doc.id)) {
          console.warn(`Document with ID "${doc.id}" already exists in "${this.storeName}". Ignoring.`);
          return; // Exit the method
      }
      this.documents.push(doc);
      console.log(`Document "${doc.id}" added to "${this.storeName}". Total: ${this.documents.length}`);
    }
  
    getDocumentById(id: string): BasicDocument | undefined {
      // Find the document with the matching ID (using an array method)
      const foundDoc = this.documents.find(doc => doc.id === id);
      if (foundDoc) {
        console.log(`Document "${id}" found in "${this.storeName}".`);
      } else {
        console.log(`Document "${id}" not found in "${this.storeName}".`);
      }
      return foundDoc; // Returns the document object or undefined
    }
  
    listDocumentIds(): string[] {
      return this.documents.map(doc => doc.id); // `map` creates a new array of IDs
    }
}
  
// --- Instantiation ---
// Creating an object (instance) from the class blueprint using `new`.
console.log("\n--- Instantiating Classes ---");
const mainStore = new DocumentStore("Main Corpus"); // Calls the constructor with "Main Corpus"
const tempStore = new DocumentStore("Temporary Uploads");

// --- Using Instances ---
mainStore.addDocument({ id: "doc-001", text: "Content for document 1" });
mainStore.addDocument({ id: "doc-002", text: "Some more content here" });
mainStore.addDocument({ id: "doc-001", text: "Attempt to add duplicate" }); // Should warn

tempStore.addDocument({ id: "temp-abc", text: "Temporary file data" });

const found = mainStore.getDocumentById("doc-002");
const notFound = mainStore.getDocumentById("doc-999");

console.log("IDs in main store:", mainStore.listDocumentIds());
console.log("IDs in temp store:", tempStore.listDocumentIds());



// ==========================================================================
// 2. Access Modifiers: public, private, protected
// ==========================================================================
console.log("\n--- 2. Access Modifiers ---");

// Control visibility and accessibility of class members (properties, methods).
// Helps with *encapsulation*: hiding internal implementation details.

// - `public`: (Default) Accessible from anywhere (inside the class, outside the class, by subclasses).
// - `private`: Accessible only from *within* the defining class itself. Not by instances outside or subclasses.
// - `protected`: Accessible from within the defining class AND from classes that *inherit* from it (subclasses).

class QueryProcessor {
    public queryCount: number = 0; // Explicitly public (same as default)
    private apiKey: string;         // Accessible only inside of QueryProcessor class
    protected processingEngine: string = "v1-standard"; // Accessible here and in subclasses
  
    constructor(apiKey: string) {
      this.apiKey = this.sanitizeKey(apiKey); // Can access private member inside the class
    }
  
    // Public method (accessible outside)
    public processQuery(query: string): void {
      this.logProcessing(query); // Can call protected method
      this.queryCount++;
      console.log(`Processing query #${this.queryCount} using engine ${this.processingEngine}: "${query}"`);
      // Simulate using the API key internally
      console.log(`  (Using sanitized key ending in: ...${this.apiKey.slice(-4)})`);
    }
  
    // Private method (accessible only inside this class)
    private sanitizeKey(key: string): string {
      // Example: basic trimming or validation (real sanitization is more complex)
      return key.trim();
    }
  
    // Protected method (accessible here and in subclasses)
    protected logProcessing(query: string): void {
      console.log(`[LOG] Preparing to process query: "${query.substring(0, 20)}..."`);
    }
}
  
const processor = new QueryProcessor("  xyz-123456789  ");
processor.processQuery("What is encapsulation?");
processor.processQuery("How do classes work?");

console.log(`Total queries processed: ${processor.queryCount}`); // OK: queryCount is public

// Access modifier enforcement by TypeScript (compile-time errors):
// console.log(processor.apiKey); // Error: Property 'apiKey' is private and only accessible within class 'QueryProcessor'.
// processor.sanitizeKey("abc"); // Error: Property 'sanitizeKey' is private...
// processor.logProcessing("test"); // Error: Property 'logProcessing' is protected... (can't access protected from outside)



// ==========================================================================
// 3. Readonly Properties in Classes
// ==========================================================================
console.log("\n--- 3. Readonly Properties ---");

// Properties marked `readonly` can only be assigned during initialization (at declaration or in the constructor).

class SystemConfig {
    readonly systemId: string; // Must be initialized
    public logLevel: string = "info";
  
    constructor(id: string) {
      this.systemId = id; // Allowed to assign here
    }
  
    setLogLevel(level: string): void {
      this.logLevel = level; // Allowed, logLevel is not readonly
    }
  
    // changeId(newId: string): void {
    //   this.systemId = newId; // Error: Cannot assign to 'systemId' because it is a read-only property.
    // }
}
  
const sysConfig = new SystemConfig("rag-system-prod");
console.log(`System ID (readonly): ${sysConfig.systemId}`);
// sysConfig.systemId = "new-id"; // Error: Cannot assign to 'systemId'...



// ==========================================================================
// 4. Static Properties and Methods (Briefly)
// ==========================================================================
console.log("\n--- 4. Static Members ---");

// `static` members belong to the *class itself*, not to instances of the class.
// Accessed using ClassName.memberName. Useful for utility functions or constants related to the class.

class VectorDBUtils {
    static readonly DEFAULT_DIMENSION = 768; // Static property
  
    static normalizeVector(vector: number[]): number[] { // Static method
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      if (magnitude === 0) return vector;
      return vector.map(val => val / magnitude);
    }
}
  
console.log(`Default vector dimension: ${VectorDBUtils.DEFAULT_DIMENSION}`);
const myVector = [1, 2, 3];
const normalized = VectorDBUtils.normalizeVector(myVector);
console.log(`Normalized vector: [${normalized.map(n => n.toFixed(2)).join(', ')}]`);

// Cannot access static members via an instance:
// const utilsInstance = new VectorDBUtils(); // Can't instantiate if no constructor/methods need `this`
// console.log(utilsInstance.DEFAULT_DIMENSION); // Error: Property 'DEFAULT_DIMENSION' does not exist on type 'VectorDBUtils'. It is a static member...



// ==========================================================================
// 5. Parameter Properties (Shorthand Constructor Initialization)
// ==========================================================================
console.log("\n--- 5. Parameter Properties ---");

// TypeScript offers a concise way to declare and initialize class properties
// directly from constructor parameters using access modifiers or `readonly`.

// --- Standard Way ---
class DataPointStandard {
    public id: string;
    public value: number;
    private source: string; // Let's make this private
    public status: string;

    constructor(id: string, value: number, source: string, status: string = 'active') { // Default value for status parameter
        this.id = id;
        this.value = value;
        this.source = source;
        this.status = status; // Assign status (will use default if not provided)
    }

    display() {
        // We can access private `source` inside the class
        console.log(`Standard - ID: ${this.id}, Value: ${this.value}, Source: ${this.source}, Status: ${this.status}`);
    }
}

// --- Using Parameter Properties (Shorthand) ---
class DataPointShorthand {
    // By adding access modifiers (or readonly), TypeScript automatically:
    // 1. Declares properties with the same name and type.
    // 2. Assigns the constructor argument value to the property.
    constructor(
        public id: string,
        public value: number,
        private source: string, // Declares and initializes private property `source`
        readonly timestamp: Date, // Declares and initializes public readonly property `timestamp`
        public status: string = 'active' // Added status property with default value directly here
    ) {
        // No `this.id = id;` etc. needed here!
        // Constructor body can still contain other logic if required.
        console.log(`Shorthand DataPoint created from source: ${this.source}`); // Can access private `source` here
    }

    display() {
        // We can access private `source` inside the class
        console.log(`Shorthand - ID: ${this.id}, Value: ${this.value}, Source: ${this.source}, Time: ${this.timestamp.toISOString()}, Status: ${this.status}`);
    }
}

// --- Usage ---
console.log("\nUsing defaults for status:");
const standardPointDefault = new DataPointStandard("dp-01", 100, "sensor-A");
standardPointDefault.display(); // Shows Status: active

const shorthandPointDefault = new DataPointShorthand("dp-02", 200, "sensor-B", new Date());
shorthandPointDefault.display(); // Shows Status: active
console.log(`Shorthand Point ID: ${shorthandPointDefault.id}`); // OK: public
// console.log(shorthandPointDefault.source); // Error: Property 'source' is private...
// shorthandPointDefault.timestamp = new Date(); // Error: Cannot assign to 'timestamp' because it is a read-only property.

console.log("\nOverriding default status:");
const standardPointOverride = new DataPointStandard("dp-03", 150, "sensor-C", "inactive");
standardPointOverride.display(); // Shows Status: inactive

const shorthandPointOverride = new DataPointShorthand("dp-04", 250, "sensor-D", new Date(), "pending");
shorthandPointOverride.display(); // Shows Status: pending


// Which to use?
// - Parameter properties are idiomatic TypeScript and reduce boilerplate. Highly recommended for simple initialization.
// - The standard way might be slightly clearer if:
//    - the constructor does complex logic beyond simple assignment (validation, calculation, method calls before assignment),
//    - You need different names for the constructor parameter and the class property.
// Both achieve the same result. Many codebases use parameter properties extensively.




// ==========================================================================
// 6. RAG Context Example: `RAGSystem` Class
// ==========================================================================
console.log("\n--- 6. RAG Context Example ---");

interface RAGConfigOptions {
    model: string;
    retrievalTopK: number;
}
  
class RAGSystem {
    private config: RAGConfigOptions;
    private documentStore: DocumentStore; // Using the class defined earlier
    private queryCount: number = 0;
    public readonly systemId: string;
  
    constructor(id: string, config: RAGConfigOptions, store: DocumentStore) {
      this.systemId = id;
      console.log(`Initializing RAG System: ${this.systemId}`);
      this.config = config;
      this.documentStore = store;
    }
  
    public addDocument(doc: BasicDocument): void {
      this.documentStore.addDocument(doc); // Delegate to the DocumentStore instance
    }
  
    public async processQuery(query: string): Promise<string> { // Using Promise/async for later modules
      this.queryCount++;
      console.log(`[${this.systemId}] Processing query #${this.queryCount}: "${query}"`);
  
      // 1. Retrieve documents (Simplified)
      const allDocIds = this.documentStore.listDocumentIds();
      console.log(`  Retrieving from ${allDocIds.length} documents... (Top K: ${this.config.retrievalTopK})`);
      // Simulate retrieval based on topK
      const contextDocs = allDocIds
          .slice(0, this.config.retrievalTopK) // Take the first K IDs (very basic simulation)
          .map(id => this.documentStore.getDocumentById(id))
          .filter(doc => doc !== undefined) as BasicDocument[]; // Type assertion needed or better filtering
  
      // 2. Format context
      const context = this.formatContext(contextDocs);
      console.log(`  Generated context:\n"${context.substring(0, 100)}..."`);
  
      // 3. Generate response (Simplified simulation)
      const response = `Response for "${query}" based on ${contextDocs.length} docs using model ${this.config.model}.`;
      console.log(`  Generated response.`);
  
      // Pretend async operation takes time
      await new Promise(resolve => setTimeout(resolve, 50)); // Simulate async delay
  
      return response;
    }
  
    // Private helper method
    private formatContext(docs: BasicDocument[]): string {
      if (docs.length === 0) {
        return "No relevant context found.";
      }
      return docs.map(doc => `[${doc.id}] ${doc.text}`).join("\n---\n");
    }
}
  
// Setup and use the RAGSystem
const ragStore = new DocumentStore("Production RAG Store");
ragStore.addDocument({ id: "rag-001", text: "Retrieval-Augmented Generation enhances LLMs." });
ragStore.addDocument({ id: "rag-002", text: "It combines retrieval with generation." });
ragStore.addDocument({ id: "rag-003", text: "Key components include retriever and generator." });

const ragConfig: RAGConfigOptions = { model: "SuperAI-v5", retrievalTopK: 2 };
const ragSystemInstance = new RAGSystem("RAG-SYS-01", ragConfig, ragStore);

// Run a query (using .then because processQuery is async - more on this later!)
ragSystemInstance.processQuery("Summarize RAG")
    .then(response => {
      console.log("\nFinal RAG Response:", response);
    })
    .catch(error => console.error("Error processing query:", error));
  
  


// --- End of Module 8 ---