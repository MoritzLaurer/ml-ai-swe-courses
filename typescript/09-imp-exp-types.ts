// 09-modules/types.ts

/**
 * This file defines and exports shared types and interfaces
 * to be used by other modules in the project.
 */

// Using Named Exports: Exporting multiple specific items by name.
export type DocumentID = string;
export type ConfidenceScore = number;

export interface BasicDocument {
  id: DocumentID;
  text: string;
}

export interface RetrievedChunk extends BasicDocument {
  // Inherits id, text from BasicDocument using interface extension
  readonly chunkId: string;
  score: ConfidenceScore;
  metadata?: object;
}

// You can also group exports at the end (alternative syntax)
// const MAX_TOKENS = 4096;
// const TIMEOUT_MS = 5000;
// export { MAX_TOKENS, TIMEOUT_MS };

// Note: No 'default' export is used in this file.