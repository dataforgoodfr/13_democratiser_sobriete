<script lang="ts">
	import {
		Conversation,
		ConversationContent,
		ConversationEmptyState,
		ConversationScrollButton
	} from '$lib/components/ai-elements/conversation';
	import {
		PromptInput,
		PromptInputTextarea,
		PromptInputSubmit
	} from '$lib/components/ai-elements/prompt-input';
	import { MessageSquare } from 'lucide-svelte';
	import type { ChatMessage, ChatStatus } from '$lib/types';
	import ChatMessageItem from './ChatMessageItem.svelte';
	import { Loader } from '../ai-elements/loader';

	interface Props {
		messages: ChatMessage[];
		status: ChatStatus;
		selectedMessageIndex: number | null;
		onSelectMessage: (index: number) => void;
		onSubmit: (input: string) => void;
	}

	let { messages, status, selectedMessageIndex, onSelectMessage, onSubmit }: Props = $props();
	let input = $state('');

	function handleSubmit(message: { text?: string }, event: SubmitEvent) {
		event.preventDefault();
		if (!input.trim() || status === 'streaming') return;
		onSubmit(input);
		input = '';
	}
</script>

<div class="flex h-full w-1/2 flex-col border-r">
	<div class="flex-1 overflow-y-auto p-6">
		<Conversation class="h-full">
			<ConversationContent>
				{#if messages.length === 0}
					<ConversationEmptyState
						title="Start a conversation"
						description="Ask about sufficiency policies and research"
					>
						{#snippet icon()}
							<MessageSquare class="size-12" />
						{/snippet}
					</ConversationEmptyState>
				{:else}
					<div class="space-y-4">
						{#each messages as message, messageIndex}
							<ChatMessageItem
								{message}
								enableCopy={status === 'idle'}
								isSelected={selectedMessageIndex === messageIndex}
								onSelect={() => onSelectMessage(messageIndex)}
							/>
						{/each}
						{#if status === 'submitted'}
							<Loader size={20} class="mx-auto w-full" />
						{/if}
					</div>
				{/if}
			</ConversationContent>
			<ConversationScrollButton />
		</Conversation>
	</div>

	<div class="border-t p-4">
		<PromptInput onSubmit={handleSubmit} class="relative w-full ">
			<div class="mx-2 flex items-center justify-between gap-2">
				<PromptInputTextarea
					bind:value={input}
					placeholder="Ask about sufficiency policies..."
					class="flex-1"
				/>
				<PromptInputSubmit {status} disabled={!input.trim()} />
			</div>
		</PromptInput>
	</div>
</div>
