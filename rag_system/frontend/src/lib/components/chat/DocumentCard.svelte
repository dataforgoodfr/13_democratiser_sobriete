<script lang="ts">
	import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-svelte';
	import type { Document, Publication, Chunk } from '$lib/types';
	import { isEvidenceChunk, isPublication } from '$lib/types';

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

<div class="rounded-lg border bg-card">
	<button
		type="button"
		class="flex w-full items-start justify-between gap-2 p-4 text-left transition-colors hover:bg-muted/30"
		onclick={onToggle}
	>
		<div class="min-w-0 flex-1">
			<div class="mb-1 flex items-center gap-2">
				<span
					class="inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground"
				>
					{index + 1}
				</span>
				<h3 class="truncate text-sm font-medium">
					{#if pub}
						{pub.title || 'Untitled Document'}
					{:else}
						Doc {index + 1}
					{/if}
				</h3>
			</div>
			{#if pub && pub.authors && pub.authors.length > 0}
				<p class="truncate text-xs text-muted-foreground">
					{pub.authors.slice(0, 3).join(', ')}{pub.authors.length > 3 ? ' et al.' : ''}
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
		<div class="border-t px-4 pb-4">
			{#if pub}
				<!-- Publication mode: show abstract and chunks -->
				{#if pub.abstract}
					<div class="pt-3">
						<h4 class="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
							Abstract
						</h4>
						<p class="text-sm leading-relaxed">{pub.abstract}</p>
					</div>
				{/if}
				{#if pub.retrieved_chunks && pub.retrieved_chunks.length > 0}
					<div class="pt-3">
						<h4 class="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
							Retrieved Chunks ({pub.retrieved_chunks.length})
						</h4>
						<div class="space-y-2">
							{#each pub.retrieved_chunks as chunkItem, chunkIdx}
								<div class="rounded-md bg-muted/50 p-3">
									<div class="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
										<p>Chunk {chunkIdx + 1}</p>
										{#if isEvidenceChunk(chunkItem)}
											<span class="rounded bg-background px-2 py-0.5 font-medium text-foreground">
												{chunkItem.sentiment}
											</span>
											<span>{chunkItem.policy_label}</span>
											<span>{chunkItem.impact_category} / {chunkItem.impact_dimension}</span>
										{/if}
									</div>
									<p class="text-sm leading-relaxed">{chunkItem.text}</p>
								</div>
							{/each}
						</div>
					</div>
				{/if}
				{#if pub.url}
					<div class="mt-3 border-t pt-3">
						<a
							href={pub.url}
							target="_blank"
							rel="noopener noreferrer"
							class="mr-4 inline-flex items-center gap-1 text-xs text-primary hover:underline"
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
					<div class="mt-3 border-t pt-3">
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
					<h4 class="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
						{isEvidenceChunk(chunk) ? 'Evidence Chunk' : 'Content'}
					</h4>
					{#if isEvidenceChunk(chunk)}
						<div class="mb-2 flex flex-wrap gap-2 text-xs">
							<span class="rounded bg-background px-2 py-1 font-medium">{chunk.sentiment}</span>
							<span class="rounded bg-background px-2 py-1">{chunk.policy_label}</span>
							<span class="rounded bg-background px-2 py-1">{chunk.impact_category} / {chunk.impact_dimension}</span>
						</div>
					{/if}
					<p class="text-sm leading-relaxed">{chunk.text}</p>
				</div>
				<div class="mt-3 border-t pt-3">
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
