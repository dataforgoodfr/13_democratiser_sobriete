<script lang="ts">
	import {
		Message,
		MessageContent,
		MessageResponse
	} from '$lib/components/ai-elements/new-message';
	import type { ChatMessage } from '$lib/types';
	import { CopyIcon } from 'lucide-svelte';
	import { Action } from '../ai-elements/action';

	interface Props {
		message: ChatMessage;
		enableCopy: boolean;
		isSelected: boolean;
		onSelect: () => void;
	}

	let { message, enableCopy, isSelected, onSelect }: Props = $props();

	async function handleCopy(text: string) {
		await navigator.clipboard.writeText(text);
	}
</script>

<!-- <button
  type="button"
  class="w-full text-left {message.role === 'assistant' ? 'cursor-pointer hover:bg-muted/50 rounded-lg transition-colors' : ''} {isSelected ? 'bg-muted/70 rounded-lg' : ''}"
  onclick={onSelect}
  disabled={message.role !== 'assistant'}
> -->
<Message from={message.role}>
	<MessageContent>
		{#if message.role === 'assistant'}
			<MessageResponse content={message.content || ''} />
		{:else}
			{message.content}
		{/if}
	</MessageContent>
</Message>
{#if message.role === 'assistant' && enableCopy}
	<Action onclick={() => handleCopy(message.content)} label="Copy" tooltip="Copy to clipboard">
		<CopyIcon class="size-4" />
	</Action>
{/if}
