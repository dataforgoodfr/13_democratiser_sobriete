import type { ChatMessage, Document } from '$lib/types';

export interface StreamCallbacks {
	onDocuments: (documents: Document[]) => void;
	onContent: (content: string) => void;
	onError: (error: Error) => void;
	onDone: () => void;
}

export async function streamChatResponse(
	chatId: string,
	messages: ChatMessage[],
	callbacks: StreamCallbacks
): Promise<void> {
	try {
		const response = await fetch('/api/chat', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				chat_id: chatId,
				messages: messages.filter((m) => m.role !== 'assistant' || m.content)
			})
		});

		if (!response.ok) throw new Error('Failed to fetch response');

		const reader = response.body?.getReader();
		const decoder = new TextDecoder();

		if (!reader) throw new Error('No reader available');

		let currentEvent = 'message';
		let buffer = ''; // Buffer for incomplete chunks, to avoid issues with latency

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });

			// Process complete messages (ending with \n\n)
			const parts = buffer.split('\n\n');
			// Keep the last part in buffer (might be incomplete)
			buffer = parts.pop() || '';

			for (const part of parts) {
				if (!part.trim()) continue;

				// Each part could contain multiple lines (event + data)
				const lines = part.split('\n\n');

				for (const line of lines) {
					if (line.startsWith('event: ')) {
						currentEvent = line.slice(7).trim();
					} else if (line.startsWith('data: ')) {
						const data = line.slice(6);
						if (data === '[DONE]') break;

						if (currentEvent === 'documents') {
							try {
								const parsed = JSON.parse(data);
								callbacks.onDocuments(parsed.documents);
							} catch (e) {
								console.error('Failed to parse documents:', e);
							}
						} else {
							const content = data.replace(/<\|newline\|>/g, '\n');
							callbacks.onContent(content);
						}
						currentEvent = 'message';
					}
				}
			}
		}

		callbacks.onDone();
	} catch (error) {
		callbacks.onError(error instanceof Error ? error : new Error(String(error)));
	}
}
