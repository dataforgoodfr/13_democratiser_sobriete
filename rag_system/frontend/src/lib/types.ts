export interface Chunk {
  openalex_id: string;
  chunk_idx: number;
  text: string;
  retrieved_rank: number;
}

export interface Publication {
  openalex_id: string;
  doi: string | null;
  title: string;
  abstract: string;
  authors: string[] | null;
  publication_year: number | null;
  url: string | null;
  retrieved_chunks: Chunk[];
}

// Union type for documents - can be either chunks or publications
export type Document = Chunk | Publication;

// Type guard to check if a document is a Publication
export function isPublication(doc: Document): doc is Publication {
  return 'retrieved_chunks' in doc;
}

// Type guard to check if a document is a Chunk
export function isChunk(doc: Document): doc is Chunk {
  return 'chunk_idx' in doc && !('retrieved_chunks' in doc);
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  documents?: Document[];
}

export type ChatStatus = "idle" | "submitted" | "streaming" | "error";
