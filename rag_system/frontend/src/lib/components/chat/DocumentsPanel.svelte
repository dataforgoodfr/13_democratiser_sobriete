<script lang="ts">
	import type { Document } from '$lib/types';
	import DocumentCard from './DocumentCard.svelte';

	interface Props {
		documents: Document[];
		isSelectedMessage: boolean;
	}

	let { documents, isSelectedMessage }: Props = $props();
	let expandedDocIndex: number | null = $state(null);

	function toggleDocExpansion(index: number) {
		expandedDocIndex = expandedDocIndex === index ? null : index;
	}
</script>

<div class="flex h-full w-1/2 flex-col">
	<div class="border-b p-4">
		<h2 class="text-lg font-semibold">📄 Retrieved Sources</h2>
		<p class="text-sm text-muted-foreground">
			{#if isSelectedMessage}
				Sources for selected message
			{:else}
				Sources for latest response
			{/if}
		</p>
	</div>

	<div class="flex-1 overflow-y-auto p-4">
		{#if documents.length === 0}
			<div class="flex h-full flex-col items-center justify-center text-muted-foreground">
				<p>No sources to display</p>
				<p class="text-sm">Sources will appear here after you ask a question</p>
			</div>
		{:else}
			<div class="space-y-3">
				{#each documents as doc, docIndex}
					<DocumentCard
						{doc}
						index={docIndex}
						expanded={expandedDocIndex === docIndex}
						onToggle={() => toggleDocExpansion(docIndex)}
					/>
				{/each}
			</div>
		{/if}
	</div>
</div>
