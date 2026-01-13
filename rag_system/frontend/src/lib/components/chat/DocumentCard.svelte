<script lang="ts">
  import { ChevronDown, ChevronUp, ExternalLink } from "lucide-svelte";
  import type { Document } from "$lib/types";

  interface Props {
    doc: Document;
    index: number;
    expanded: boolean;
    onToggle: () => void;
  }

  let { doc, index, expanded, onToggle }: Props = $props();

  function getOpenAlexUrl(id: string): string {
    return `https://openalex.org/${id}`;
  }
</script>

<div class="border rounded-lg bg-card">
  <button
    type="button"
    class="w-full p-4 text-left flex items-start justify-between gap-2 hover:bg-muted/30 transition-colors"
    onclick={onToggle}
  >
    <div class="flex-1 min-w-0">
      <div class="flex items-center gap-2 mb-1">
        <span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-medium">
          {index + 1}
        </span>
        <h3 class="font-medium text-sm truncate">
          {doc.title || "Untitled Document"}
        </h3>
      </div>
      {#if doc.authors && doc.authors.length > 0}
        <p class="text-xs text-muted-foreground truncate">
          {doc.authors.slice(0, 3).join(", ")}{doc.authors.length > 3 ? " et al." : ""}
          {#if doc.publication_year}
            ({doc.publication_year})
          {/if}
        </p>
      {/if}
    </div>
    <div class="flex-shrink-0">
      {#if expanded}
        <ChevronUp class="size-5 text-muted-foreground" />
      {:else}
        <ChevronDown class="size-5 text-muted-foreground" />
      {/if}
    </div>
  </button>
  
  {#if expanded}
    <div class="px-4 pb-4 border-t">
      <div class="pt-3">
        <h4 class="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Abstract</h4>
        <p class="text-sm leading-relaxed">{doc.text}</p>
      </div>
      <div class="mt-3 pt-3 border-t">
        <a
          href={getOpenAlexUrl(doc.openalex_id)}
          target="_blank"
          rel="noopener noreferrer"
          class="inline-flex items-center gap-1 text-xs text-primary hover:underline"
        >
          <ExternalLink class="size-3" />
          View on OpenAlex
        </a>
      </div>
    </div>
  {/if}
</div>
