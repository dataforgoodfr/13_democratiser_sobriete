export interface Document {
  openalex_id: string;
  title: string | null;
  text: string;
  authors: string[] | null;
  publication_year: number | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  documents?: Document[];
}

export type ChatStatus = "idle" | "submitted" | "streaming" | "error";
