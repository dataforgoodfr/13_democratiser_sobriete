<script lang="ts">
	import { page } from '$app/state';
	import { ChatPanel, DocumentsPanel } from '$lib/components/chat';
	import PolicyAnalysisPanel from '$lib/components/chat/PolicyAnalysisPanel.svelte';
	import WelcomeModal from '$lib/components/WelcomeModal.svelte';
	import NavBar from '$lib/components/NavBar.svelte';
	import { streamChatResponse } from '$lib/services/chatService';
	import type { ChatMessage, ChatStatus, Document, Policy, RetrievalStep } from '$lib/types';
	import { env as publicEnv } from '$env/dynamic/public';
	import { PUBLIC_POLICY_PANEL_ENABLED } from '$env/static/public';

	const policyPanelEnabled = PUBLIC_POLICY_PANEL_ENABLED !== 'false';
	const isPolicyFirst = publicEnv.PUBLIC_RAG_PIPELINE === 'policy';

	let chatId = $state(crypto.randomUUID());
	let messages: ChatMessage[] = $state([]);
	let status: ChatStatus = $state('idle');
	let selectedMessageIndex: number | null = $state(null);
	let activeTab: 'chat' | 'sources' | 'policies' = $state('chat');
	let retrievalStep: RetrievalStep = $state(null);
	const isAuthenticated = $derived(!!page.data.session?.user);
	const userName = $derived(page.data.session?.user?.name || page.data.session?.user?.email || null);
	const userEmail = $derived(page.data.session?.user?.email ?? null);

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

	// same for policies
	let displayedPolicies = $derived.by((): Policy[] => {
		if (selectedMessageIndex !== null) {
			return messages[selectedMessageIndex]?.policies || [];
		}
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === 'assistant' && messages[i].policies) {
				return messages[i].policies || [];
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
		retrievalStep = null;
	}

	async function handleSubmit(input: string) {
		const userMessage: ChatMessage = { role: 'user', content: input };
		messages = [...messages, userMessage];
		status = 'submitted';

		// Add placeholder for assistant response
		const assistantMessage: ChatMessage = { role: 'assistant', content: '', documents: [], policies: [] };
		messages = [...messages, assistantMessage];
		selectedMessageIndex = null;

		await streamChatResponse(chatId, userEmail, messages, {
			onDocuments: (documents: Document[]) => {
				messages = messages.map((msg, i) =>
					i === messages.length - 1 ? { ...msg, documents } : msg
				);
			},
			onPolicies: (policies) => {
				messages = messages.map((msg, i) =>
					i === messages.length - 1 ? { ...msg, policies } : msg
				);
			},
			onStatus: (step) => {
				retrievalStep = step;
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
				retrievalStep = null;
			},
			onDone: () => {
				status = 'idle';
				retrievalStep = null;
			}
		});
	}
</script>

<div class="flex h-screen w-screen flex-col bg-background">
	<NavBar onReset={handleReset} {chatId} {userName} />
	
	<!-- Tab buttons for mobile only -->
	<div class="flex border-b md:hidden">
		<button
			class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'chat'
				? 'border-b-2 border-primary text-foreground'
				: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => (activeTab = 'chat')}
		>
			💬 Chat
		</button>
		{#if policyPanelEnabled && isPolicyFirst}
			<button
				class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'policies'
					? 'border-b-2 border-primary text-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (activeTab = 'policies')}
			>
				🏛️ Policies
				{#if displayedPolicies.length > 0}
					<span class="ml-1 rounded-full bg-primary px-2 py-0.5 text-xs text-primary-foreground">
						{displayedPolicies.length}
					</span>
				{/if}
			</button>
		{/if}
		<button
			class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'sources'
				? 'border-b-2 border-primary text-foreground'
				: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => (activeTab = 'sources')}
		>
			📄 Sources
			{#if displayedDocuments.length > 0}
				<span class="ml-1 rounded-full bg-primary px-2 py-0.5 text-xs text-primary-foreground">
					{displayedDocuments.length}
				</span>
			{/if}
		</button>
		{#if policyPanelEnabled && !isPolicyFirst}
		<button
			class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'policies'
				? 'border-b-2 border-primary text-foreground'
				: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => (activeTab = 'policies')}
		>
			🏛️ Policies
			{#if displayedPolicies.length > 0}
				<span class="ml-1 rounded-full bg-primary px-2 py-0.5 text-xs text-primary-foreground">
					{displayedPolicies.length}
				</span>
			{/if}
		</button>
		{/if}
	</div>

	<div class="flex flex-1 overflow-hidden">
		<!-- Chat panel: full width on mobile when chat tab active, half width on desktop -->
		<div class="w-full md:w-1/2 {activeTab === 'chat' ? 'block' : 'hidden md:block'}">
			<ChatPanel
				{messages}
				{status}
				{selectedMessageIndex}
				{retrievalStep}
				onSelectMessage={handleSelectMessage}
				onSubmit={handleSubmit}
			/>
		</div>

		<!-- Right column: sources + optional policy panel -->
		<div class="{activeTab === 'chat' ? 'hidden' : 'flex'} w-full flex-col overflow-hidden md:flex md:w-1/2">
			{#if policyPanelEnabled && isPolicyFirst}
				<div
					class="{activeTab !== 'policies' ? 'hidden md:flex' : 'flex'} flex-1 flex-col overflow-hidden min-h-0 md:flex-none md:h-1/2"
				>
					<PolicyAnalysisPanel policies={displayedPolicies} />
				</div>
				<div
					class="{activeTab !== 'sources' ? 'hidden md:flex' : 'flex'} flex-1 flex-col overflow-hidden min-h-0 border-t md:flex-none md:h-1/2"
				>
					<DocumentsPanel
						documents={displayedDocuments}
						isSelectedMessage={selectedMessageIndex !== null}
					/>
				</div>
			{:else}
				<div
					class="{activeTab !== 'sources' ? 'hidden md:flex' : 'flex'} flex-1 flex-col overflow-hidden min-h-0 {policyPanelEnabled ? 'md:flex-none md:h-1/2' : ''}"
				>
					<DocumentsPanel
						documents={displayedDocuments}
						isSelectedMessage={selectedMessageIndex !== null}
					/>
				</div>

				{#if policyPanelEnabled}
				<div
					class="{activeTab !== 'policies' ? 'hidden md:flex' : 'flex'} flex-1 flex-col overflow-hidden min-h-0 border-t md:flex-none md:h-1/2"
				>
					<PolicyAnalysisPanel policies={displayedPolicies} />
				</div>
				{/if}
			{/if}
		</div>
	</div>
</div>

{#if !isAuthenticated}
	<WelcomeModal />
{/if}
