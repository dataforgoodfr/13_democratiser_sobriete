import type { ChatMessage, Document } from '$lib/types';

const API_URL = 'http://localhost:8000/api/chat';

export interface StreamCallbacks {
	onDocuments: (documents: Document[]) => void;
	onContent: (content: string) => void;
	onError: (error: Error) => void;
	onDone: () => void;
}

export async function streamChatResponse(
	messages: ChatMessage[],
	callbacks: StreamCallbacks
): Promise<void> {
	try {
		const response = await fetch(API_URL, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				messages: messages.filter((m) => m.role !== 'assistant' || m.content)
			})
		});

		if (!response.ok) throw new Error('Failed to fetch response');

		const reader = response.body?.getReader();
		const decoder = new TextDecoder();

		if (!reader) throw new Error('No reader available');

		let currentEvent = 'message';

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			const chunk = decoder.decode(value, { stream: true });
			const lines = chunk.split('\n');

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

		callbacks.onDone();
	} catch (error) {
		callbacks.onError(error instanceof Error ? error : new Error(String(error)));
	}
}
