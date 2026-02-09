<script lang="ts">
	import { SquarePenIcon, MessageSquareIcon } from '@lucide/svelte';
	import WSLLogoFull from '$lib/assets/wsl_logo.svg';
	import WSLLogoIcon from '$lib/assets/wsl_icon.svg';
	import { AppBar } from '@skeletonlabs/skeleton-svelte';
	import FeedbackModal from './FeedbackModal.svelte';
	import { PUBLIC_SHOW_FULL_NAVBAR } from '$env/static/public';

	interface Props {
		onReset?: () => void;
		chatId?: string;
	}

	let { onReset, chatId }: Props = $props();

	let showFeedbackModal = $state(false);
	const showFullNavbar = PUBLIC_SHOW_FULL_NAVBAR === 'true';
</script>

{#if showFullNavbar}
	<!-- Full responsive navbar -->
	<AppBar class="bg-wsl-yellow p-2 md:p-4">
		<AppBar.Toolbar class="flex flex-wrap w-full items-center gap-2 md:gap-6">
			<AppBar.Lead>
				<img class="h-8 sm:h-12 hidden sm:block" alt="World Sufficiency Lab logo" src={WSLLogoFull} />
				<img class="h-8 sm:h-12 sm:hidden" alt="World Sufficiency Lab logo" src={WSLLogoIcon} />
			</AppBar.Lead>
			<AppBar.Headline>
				<p class="text-xl sm:text-3xl font-bold">Chat Sufficiency</p>
			</AppBar.Headline>
			<AppBar.Trail class="sm:ml-auto flex gap-2">
				<button
					class="flex items-center gap-2 rounded-lg bg-white px-2 py-1.5 md:px-4 md:py-2 text-black hover:bg-gray-100"
					onclick={() => (showFeedbackModal = true)}
					title="Feedback"
				>
					<MessageSquareIcon size={20} />
					<span>Feedback</span>
				</button>
				<button
					class="flex items-center gap-2 rounded-lg bg-black px-2 py-1.5 md:px-4 md:py-2 text-white hover:font-semibold"
					onclick={onReset}
					title="New chat"
				>
					<SquarePenIcon size={20} />
					<span>New chat</span>
				</button>
			</AppBar.Trail>
		</AppBar.Toolbar>
	</AppBar>
{:else}
	<!-- Minimal header with just buttons -->
	<div class="flex items-center justify-end gap-2 border-b bg-background p-2">
		<button
			class="flex items-center gap-2 rounded-lg border bg-card px-2 py-1.5 md:px-4 md:py-2 text-card-foreground hover:bg-accent"
			onclick={() => (showFeedbackModal = true)}
			title="Feedback"
		>
			<MessageSquareIcon size={20} />
			<span class="hidden sm:inline">Feedback</span>
		</button>
		<button
			class="flex items-center gap-2 rounded-lg bg-primary px-2 py-1.5 md:px-4 md:py-2 text-primary-foreground hover:bg-primary/90"
			onclick={onReset}
			title="New chat"
		>
			<SquarePenIcon size={20} />
			<span class="hidden sm:inline">New chat</span>
		</button>
	</div>
{/if}

{#if showFeedbackModal}
	<FeedbackModal {chatId} onClose={() => (showFeedbackModal = false)} />
{/if}
