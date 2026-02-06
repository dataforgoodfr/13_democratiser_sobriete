<script lang="ts">
	import { ChatPanel, DocumentsPanel } from '$lib/components/chat';
	import DisclaimerModal from '$lib/components/DisclaimerModal.svelte';
	import NavBar from '$lib/components/NavBar.svelte';
	import { streamChatResponse } from '$lib/services/chatService';
	import type { ChatMessage, ChatStatus, Document } from '$lib/types';

	let chatId = $state(crypto.randomUUID());
	let messages: ChatMessage[] = $state([]);
	let status: ChatStatus = $state('idle');
	let selectedMessageIndex: number | null = $state(null);
	let showDisclaimer = $state(true);

	// Get documents for the selected message (or last assistant message if none selected)
	let displayedDocuments = $derived.by(() => {
		if (selectedMessageIndex !== null) {
			return messages[selectedMessageIndex]?.documents || [];
		}
		// Find the last assistant message with documents
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === 'assistant' && messages[i].documents) {
				return messages[i].documents || [];
			}
		}
		return [];
	});

	function handleSelectMessage(index: number) {
		if (messages[index].role === 'assistant') {
			selectedMessageIndex = selectedMessageIndex === index ? null : index;
		}
	}

	function handleReset() {
		chatId = crypto.randomUUID();
		messages = [];
		selectedMessageIndex = null;
		status = 'idle';
	}

	async function handleSubmit(input: string) {
		const userMessage: ChatMessage = { role: 'user', content: input };
		messages = [...messages, userMessage];
		status = 'submitted';

		// Add placeholder for assistant response
		const assistantMessage: ChatMessage = { role: 'assistant', content: '', documents: [] };
		messages = [...messages, assistantMessage];
		selectedMessageIndex = null;

		await streamChatResponse(chatId, messages, {
			onDocuments: (documents: Document[]) => {
				messages = messages.map((msg, i) =>
					i === messages.length - 1 ? { ...msg, documents } : msg
				);
			},
			onContent: (content: string) => {
				status = 'streaming';
				messages = messages.map((msg, i) =>
					i === messages.length - 1 ? { ...msg, content: msg.content + content } : msg
				);
			},
			onError: (error: Error) => {
				console.error('Error:', error);
				messages = messages.map((msg, i) =>
					i === messages.length - 1 ? { ...msg, content: 'Error: Failed to get response' } : msg
				);
				status = 'idle';
			},
			onDone: () => {
				status = 'idle';
			}
		});
	}
</script>

<div class="flex h-screen w-screen flex-col bg-background">
	<NavBar onReset={handleReset} {chatId} />
	<div class="flex flex-1 overflow-hidden">
		<ChatPanel
			{messages}
			{status}
			{selectedMessageIndex}
			onSelectMessage={handleSelectMessage}
			onSubmit={handleSubmit}
		/>

		<DocumentsPanel
			documents={displayedDocuments}
			isSelectedMessage={selectedMessageIndex !== null}
		/>
	</div>
</div>

{#if showDisclaimer}
	<DisclaimerModal onClose={() => (showDisclaimer = false)} />
{/if}
