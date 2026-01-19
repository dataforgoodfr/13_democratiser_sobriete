<script lang="ts">
	import { XIcon } from '@lucide/svelte';
	import { submitFeedback } from '$lib/services';

	interface Props {
		chatId?: string;
		onClose: () => void;
	}

	let { chatId, onClose }: Props = $props();

	let feedbackContent = $state('');
	let isSubmitting = $state(false);
	let feedbackStatus: 'idle' | 'success' | 'error' = $state('idle');

	async function handleSubmit() {
		if (!feedbackContent.trim()) return;

		isSubmitting = true;
		try {
			const response = await submitFeedback({
				chatId,
				content: feedbackContent
			});

			if (response.status === 'success') {
				feedbackStatus = 'success';
				setTimeout(() => {
					onClose();
				}, 1500);
			} else {
				feedbackStatus = 'error';
			}
		} catch (error) {
			console.error('Error submitting feedback:', error);
			feedbackStatus = 'error';
		} finally {
			isSubmitting = false;
		}
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
	class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
	onclick={onClose}
>
	<div
		class="mx-4 w-full max-w-lg rounded-lg bg-white p-6 shadow-xl"
		onclick={(e) => e.stopPropagation()}
	>
		<div class="mb-4 flex items-center justify-between">
			<h2 class="text-xl font-bold">Send Feedback</h2>
			<button class="text-gray-500 hover:text-gray-700" onclick={onClose}>
				<XIcon size={20} />
			</button>
		</div>

		{#if feedbackStatus === 'success'}
			<div class="rounded-lg bg-green-100 p-4 text-green-800">
				Thank you for your feedback!
			</div>
		{:else}
			<textarea
				class="mb-4 h-40 w-full resize-none rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:outline-none"
				placeholder="Share your thoughts, suggestions, or report issues..."
				bind:value={feedbackContent}
				disabled={isSubmitting}
			></textarea>

			{#if feedbackStatus === 'error'}
				<div class="mb-4 rounded-lg bg-red-100 p-3 text-red-800">
					Failed to submit feedback. Please try again.
				</div>
			{/if}

			<div class="flex justify-end gap-3">
				<button
					class="rounded-lg px-4 py-2 text-gray-600 hover:bg-gray-100"
					onclick={onClose}
					disabled={isSubmitting}
				>
					Cancel
				</button>
				<button
					class="rounded-lg bg-black px-4 py-2 text-white hover:bg-gray-800 disabled:opacity-50"
					onclick={handleSubmit}
					disabled={isSubmitting || !feedbackContent.trim()}
				>
					{isSubmitting ? 'Submitting...' : 'Submit'}
				</button>
			</div>
		{/if}
	</div>
</div>
