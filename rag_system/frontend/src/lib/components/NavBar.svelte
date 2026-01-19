<script lang="ts">
	import { SquarePenIcon, MessageSquareIcon } from '@lucide/svelte';
	import WSLLogo from '$lib/assets/wsl_logo.svg';
	import { AppBar } from '@skeletonlabs/skeleton-svelte';
	import FeedbackModal from './FeedbackModal.svelte';

	interface Props {
		onReset?: () => void;
		chatId?: string;
	}

	let { onReset, chatId }: Props = $props();

	let showFeedbackModal = $state(false);
</script>

<AppBar class="bg-wsl-yellow p-4">
	<AppBar.Toolbar class="flex w-full items-center gap-6">
		<AppBar.Lead>
			<img class="h-12" alt="World Sufficiency Lab logo" src={WSLLogo} />
		</AppBar.Lead>
		<AppBar.Headline>
			<p class="text-3xl font-bold">Chat Sufficiency</p>
		</AppBar.Headline>
		<AppBar.Trail class="ml-auto flex gap-2">
			<button
				class="flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-black hover:bg-gray-100"
				onclick={() => (showFeedbackModal = true)}
			>
				<MessageSquareIcon size={20} />
				<span>Feedback</span>
			</button>
			<button
				class="flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-white hover:font-semibold"
				onclick={onReset}
			>
				<SquarePenIcon size={20} />
				<span>New chat</span>
			</button>
		</AppBar.Trail>
	</AppBar.Toolbar>
</AppBar>

{#if showFeedbackModal}
	<FeedbackModal {chatId} onClose={() => (showFeedbackModal = false)} />
{/if}
