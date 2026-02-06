<script lang="ts">
	import { XIcon, FlaskConicalIcon } from '@lucide/svelte';

	interface Props {
		onClose: () => void;
	}

	let { onClose }: Props = $props();

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onClose();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
	class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
	onclick={onClose}
>
	<div
		class="mx-4 w-full max-w-md rounded-xl bg-white p-6 shadow-2xl"
		onclick={(e) => e.stopPropagation()}
	>
		<div class="mb-3 flex items-start justify-between">
			<div class="flex items-center gap-2.5">
				<h2 class="text-lg font-bold text-gray-900">Welcome to Chat Sufficiency!</h2>
			</div>
			<button
				class="rounded-md p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
				onclick={onClose}
			>
				<XIcon size={18} />
			</button>
		</div>

		<p class="mb-3 text-sm leading-relaxed text-gray-600">
            This tool allows you to explore the scientific literature on sufficiency policies through a conversational interface.
		</p>

		<div class="mb-5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
			<p class="text-sm leading-relaxed text-amber-800">
				<strong>⚠️ Preview version</strong> — We're still working on the retrieval pipeline
                to build the most precise context possible for the model from the literature.
                Also, as any AI and despite our best efforts, Chat Sufficiency can make mistakes.
				We encourage you to read the cited works.
			</p>
		</div>

		<button
			class="w-full rounded-lg bg-black px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-800"
			onclick={onClose}
		>
			Understood. Let's get started!
		</button>
	</div>
</div>
