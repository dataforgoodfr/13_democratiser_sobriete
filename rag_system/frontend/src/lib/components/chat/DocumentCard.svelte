<script lang="ts">
  import { ChevronDown, ChevronUp, ExternalLink } from "lucide-svelte";
  import type { Document, Publication, Chunk } from "$lib/types";
  import { isPublication } from "$lib/types";

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

  // Determine if we're showing a publication or a chunk
  let isPub = $derived(isPublication(doc));
  // Cast for type-safe access when isPub is true
  let pub = $derived(isPub ? (doc as Publication) : null);
  let chunk = $derived(!isPub ? (doc as Chunk) : null);
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
          {#if pub}
            {pub.title || "Untitled Document"}
          {:else}
            Doc {index + 1}
          {/if}
        </h3>
      </div>
      {#if pub && pub.authors && pub.authors.length > 0}
        <p class="text-xs text-muted-foreground truncate">
          {pub.authors.slice(0, 3).join(", ")}{pub.authors.length > 3 ? " et al." : ""}
          {#if pub.publication_year}
            ({pub.publication_year})
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
      {#if pub}
        <!-- Publication mode: show abstract and chunks -->
        {#if pub.abstract}
          <div class="pt-3">
            <h4 class="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Abstract</h4>
            <p class="text-sm leading-relaxed">{pub.abstract}</p>
          </div>
        {/if}
        {#if pub.retrieved_chunks && pub.retrieved_chunks.length > 0}
          <div class="pt-3">
            <h4 class="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              Retrieved Chunks ({pub.retrieved_chunks.length})
            </h4>
            <div class="space-y-2">
              {#each pub.retrieved_chunks as chunkItem, chunkIdx}
                <div class="p-3 bg-muted/50 rounded-md">
                  <p class="text-xs text-muted-foreground mb-1">Chunk {chunkIdx + 1}</p>
                  <p class="text-sm leading-relaxed">{chunkItem.text}</p>
                </div>
              {/each}
            </div>
          </div>
        {/if}
        {#if pub.url}
          <div class="mt-3 pt-3 border-t">
            <a
              href={pub.url}
              target="_blank"
              rel="noopener noreferrer"
              class="inline-flex items-center gap-1 text-xs text-primary hover:underline mr-4"
            >
              <ExternalLink class="size-3" />
              View PDF
            </a>
            <a
              href={getOpenAlexUrl(pub.openalex_id)}
              target="_blank"
              rel="noopener noreferrer"
              class="inline-flex items-center gap-1 text-xs text-primary hover:underline"
            >
              <ExternalLink class="size-3" />
              View on OpenAlex
            </a>
          </div>
        {:else}
          <div class="mt-3 pt-3 border-t">
            <a
              href={getOpenAlexUrl(pub.openalex_id)}
              target="_blank"
              rel="noopener noreferrer"
              class="inline-flex items-center gap-1 text-xs text-primary hover:underline"
            >
              <ExternalLink class="size-3" />
              View on OpenAlex
            </a>
          </div>
        {/if}
      {:else if chunk}
        <!-- Chunk mode: just show the text -->
        <div class="pt-3">
          <h4 class="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Content</h4>
          <p class="text-sm leading-relaxed">{chunk.text}</p>
        </div>
        <div class="mt-3 pt-3 border-t">
          <a
            href={getOpenAlexUrl(chunk.openalex_id)}
            target="_blank"
            rel="noopener noreferrer"
            class="inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            <ExternalLink class="size-3" />
            View on OpenAlex
          </a>
        </div>
      {/if}
    </div>
  {/if}
</div>
