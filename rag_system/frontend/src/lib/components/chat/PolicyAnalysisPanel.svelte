<script lang="ts">
	import { PUBLIC_POLICY_COLOR_ENABLED } from '$env/static/public';
	import { isPolicySearchResult, type Policy } from '$lib/types';

	interface Props {
		policies: Policy[];
	}

	let { policies }: Props = $props();
	let expandedIndex: number | null = $state(null);

	const colorEnabled = PUBLIC_POLICY_COLOR_ENABLED !== 'false';
	const hasPolicyFirstResults = $derived(policies.some(isPolicySearchResult));

	const CLASS_ORDER: Record<string, number> = { S: 0, PS: 1, NS: 2 };

	const sortedPolicies = $derived(
		hasPolicyFirstResults
			? [...policies]
			: [...policies].sort(
					(a, b) =>
						(CLASS_ORDER[!isPolicySearchResult(a) ? a.sufficiency_class : ''] ?? 99) -
						(CLASS_ORDER[!isPolicySearchResult(b) ? b.sufficiency_class : ''] ?? 99)
				)
	);

	const CLASS_COLORS: Record<string, string> = {
		S: 'border-l-4 border-l-green-500',
		PS: 'border-l-4 border-l-yellow-400',
		NS: 'border-l-4 border-l-red-500',
	};

	const BADGE_COLORS: Record<string, string> = {
		S: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
		PS: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
		NS: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
	};

	function toggle(index: number) {
		expandedIndex = expandedIndex === index ? null : index;
	}

	function sentimentBadgeClass(sentiment: 'positive' | 'neutral' | 'negative') {
		if (sentiment === 'positive') return 'bg-emerald-100 text-emerald-800';
		if (sentiment === 'negative') return 'bg-rose-100 text-rose-800';
		return 'bg-slate-100 text-slate-800';
	}
</script>

<div class="flex h-full w-full flex-col">
	<div class="border-b p-4">
		<h2 class="text-lg font-semibold">🏛️ Policy Analysis</h2>
		{#if hasPolicyFirstResults}
			<p class="text-sm text-muted-foreground">Policies retained after relevance reranking</p>
			<p class="text-sm text-muted-foreground">Impact counts and sampled evidence help surface both pros and cons</p>
		{:else}
			<p class="text-sm text-muted-foreground">Policies identified in the retrieved context</p>
			<p class="text-sm text-muted-foreground">S = Sufficiency, PS = Possible Sufficiency, NS = Not Sufficiency</p>
		{/if}
	</div>

	<div class="flex-1 overflow-y-auto p-4">
		{#if policies.length === 0}
			<div class="flex h-full flex-col items-center justify-center text-muted-foreground">
				<p>No policies identified</p>
				<p class="text-sm">Policy analysis will appear here after you ask a question</p>
			</div>
		{:else}
			<div class="space-y-2">
			{#each sortedPolicies as policy, i}
				<div class="rounded-lg border bg-card text-card-foreground shadow-sm {(!hasPolicyFirstResults && colorEnabled && !isPolicySearchResult(policy)) ? CLASS_COLORS[policy.sufficiency_class] ?? '' : ''}">
					<!-- Header / toggle -->
					<button
						class="flex w-full items-center justify-between px-4 py-3 text-left"
						onclick={() => toggle(i)}
					>
						<div class="min-w-0">
							{#if isPolicySearchResult(policy)}
								<p class="font-medium leading-snug">{policy.policy_text}</p>
								<p class="text-xs text-muted-foreground">Cluster {policy.cluster_id} · Score {policy.rerank_score}/9</p>
							{:else}
								<span class="font-medium">{policy.cluster}</span>
							{/if}
						</div>
						<div class="flex items-center gap-2">
							{#if colorEnabled && !isPolicySearchResult(policy)}
								<span class="rounded px-1.5 py-0.5 text-xs font-bold {BADGE_COLORS[policy.sufficiency_class] ?? ''}">
									{policy.sufficiency_class}
								</span>
							{:else if isPolicySearchResult(policy)}
								<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
									{policy.count} refs
								</span>
							{/if}
							<svg
								class="h-4 w-4 flex-shrink-0 text-muted-foreground transition-transform {expandedIndex === i ? 'rotate-180' : ''}"
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
							>
								<polyline points="6 9 12 15 18 9" />
							</svg>
						</div>
						</button>

						<!-- Expanded content -->
						{#if expandedIndex === i}
							<div class="border-t px-4 pb-4 pt-3 space-y-3">
								{#if isPolicySearchResult(policy)}
									<div>
										<p class="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
											Reranker Reasoning
										</p>
										<p class="text-sm">{policy.rerank_reasoning}</p>
									</div>
									<div class="flex flex-wrap gap-2">
										<span class="rounded px-2 py-1 text-xs font-medium {sentimentBadgeClass('positive')}">
											Positive {policy.positive_count}
										</span>
										<span class="rounded px-2 py-1 text-xs font-medium {sentimentBadgeClass('neutral')}">
											Neutral {policy.neutral_count}
										</span>
										<span class="rounded px-2 py-1 text-xs font-medium {sentimentBadgeClass('negative')}">
											Negative {policy.negative_count}
										</span>
									</div>
	
									{#if policy.matched_impact_categories.length > 0}
										<div>
											<p class="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Matched Categories</p>
											<p class="text-sm">{policy.matched_impact_categories.join(', ')}</p>
										</div>
									{/if}
									{#if policy.matched_impact_dimensions.length > 0}
										<div>
											<p class="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Matched Dimensions</p>
											<p class="text-sm">{policy.matched_impact_dimensions.join(', ')}</p>
										</div>
									{/if}
								{:else}
									<div>
										<p class="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
											Sufficiency Assessment
										</p>
										<p class="text-sm">{policy.sufficiency_classification_reasoning}</p>
									</div>
								{/if}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
