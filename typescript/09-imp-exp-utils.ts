// 09-modules/utils.ts

/**
 * Defines and exports utility functions.
 */
import { RetrievedChunk } from './09-imp-exp-types.js'; // Import needed type (note .js extension)

// Exporting functions using named exports

export function calculateWordCount(text: string): number {
  if (!text) return 0;
  return text.trim().split(/\s+/).length;
}

export function formatContextForLLM(chunks: RetrievedChunk[], maxChars?: number): string {
  console.log("(Module) Formatting context...");
  let combinedText = "";
  for (const chunk of chunks) {
    const chunkText = `DocID: ${chunk.id}\nScore: ${chunk.score.toFixed(2)}\nText: ${chunk.text}\n---\n`;
    if (maxChars !== undefined && (combinedText.length + chunkText.length > maxChars)) {
      console.warn("(Module) Max character limit reached during context formatting.");
      break;
    }
    combinedText += chunkText;
  }
  return combinedText.trim();
}

// Could also have a default export if one function was primary
// export default calculateWordCount;