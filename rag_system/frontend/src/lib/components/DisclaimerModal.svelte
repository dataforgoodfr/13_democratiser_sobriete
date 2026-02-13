<script lang="ts">
	import { GraduationCapIcon, BookOpenIcon, BriefcaseIcon, LandmarkIcon, MegaphoneIcon, ScaleIcon, UserIcon } from '@lucide/svelte';

	interface Props {
		onClose: (persona: string) => void;
	}

	let { onClose }: Props = $props();

	const personas = [
		{ id: 'researcher', label: 'Researcher', icon: BookOpenIcon },
		{ id: 'student', label: 'Student', icon: GraduationCapIcon },
		{ id: 'practitioner', label: 'Practitioner', icon: BriefcaseIcon },
		{ id: 'policymaker', label: 'Policymaker', icon: LandmarkIcon },
		{ id: 'advocate', label: 'Advocate', icon: MegaphoneIcon },
		{ id: 'lobbyist', label: 'Lobbyist', icon: ScaleIcon },
		{ id: 'citizen', label: 'Just a curious citizen', icon: UserIcon },
	];
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
	class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
>
	<div
		class="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl bg-white p-6 shadow-2xl"
		onclick={(e) => e.stopPropagation()}
	>
		<div class="mb-3 flex items-start justify-between">
			<div class="flex items-center gap-2.5">
				<h2 class="text-lg font-bold text-gray-900">Welcome to Chat Sufficiency!</h2>
			</div>
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

		<div class="mb-4">
			<p class="mb-3 text-sm font-medium text-gray-700">Tell us about yourself to get started:</p>
			<div class="flex flex-wrap justify-center gap-2">
				{#each personas as persona}
					<!-- the complicated width calculation is for centering the buttons in each row -->
					<button
						class="flex w-[calc(50%-0.25rem)] sm:w-[calc(33.333%-0.35rem)] flex-col items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-3 text-sm font-medium text-gray-700 transition-all hover:border-gray-400 hover:bg-gray-50 hover:shadow-sm"
						onclick={() => onClose(persona.id)}
					>
						<persona.icon size={20} class="text-gray-500" />
						<span class="text-center text-xs">{persona.label}</span>
					</button>
				{/each}
			</div>
		</div>
	</div>
</div>
