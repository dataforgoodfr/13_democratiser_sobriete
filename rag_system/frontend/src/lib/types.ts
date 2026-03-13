export interface Chunk {
  openalex_id: string;
  chunk_idx: number;
  text: string;
  retrieved_rank: number;
}

export interface EvidenceChunk extends Chunk {
  sentiment: 'positive' | 'neutral' | 'negative';
  impact_category: string;
  impact_dimension: string;
  policy_cluster_id: string | number;
  policy_label: string;
}

export interface Publication {
  openalex_id: string;
  doi: string | null;
  title: string;
  abstract: string | null;
  authors: string[] | null;
  publication_year: number | null;
  url: string | null;
  retrieved_chunks: Array<Chunk | EvidenceChunk>;
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

export function isEvidenceChunk(chunk: Chunk | EvidenceChunk): chunk is EvidenceChunk {
  return 'sentiment' in chunk;
}

export interface PolicyImpact {
  cluster: string;
  sufficiency_class: 'S' | 'PS' | 'NS';
  sufficiency_classification_reasoning: string;
}

export interface PolicySearchResult {
  cluster_id: string | number;
  policy_text: string;
  count: number;
  retrieved_rank: number;
  retrieved_score: number | null;
  rerank_score: number;
  rerank_reasoning: string;
  matched_impact_categories: string[];
  matched_impact_dimensions: string[];
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  sampled_positive_refs: number;
  sampled_neutral_refs: number;
  sampled_negative_refs: number;
}

export type Policy = PolicyImpact | PolicySearchResult;

export function isPolicySearchResult(policy: Policy): policy is PolicySearchResult {
  return 'rerank_score' in policy;
}

export type RetrievalStep =
  | 'analyzing_query'
  | 'retrieving_sources'
  | 'analyzing_policies'
  | 'retrieving_policies'
  | 'retrieving_evidence'
  | null;

export const RETRIEVAL_STEP_LABELS: Record<NonNullable<RetrievalStep>, string> = {
  analyzing_query: 'Analyzing query…',
  retrieving_sources: 'Retrieving sources…',
  analyzing_policies: 'Analyzing policies…',
  retrieving_policies: 'Retrieving policies…',
  retrieving_evidence: 'Fetching evidence…',
};

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  documents?: Document[];
  policies?: Policy[];
}

export type ChatStatus = "idle" | "submitted" | "streaming" | "error";
