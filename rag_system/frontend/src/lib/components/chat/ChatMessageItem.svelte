<script lang="ts">
	import {
		Message,
		MessageContent,
		MessageResponse
	} from '$lib/components/ai-elements/new-message';
	import type { ChatMessage } from '$lib/types';

	interface Props {
		message: ChatMessage;
		isSelected: boolean;
		onSelect: () => void;
	}

	let { message, isSelected, onSelect }: Props = $props();
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
