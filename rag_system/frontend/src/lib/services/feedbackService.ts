export interface FeedbackRequest {
	chatId?: string;
	content: string;
}

export interface FeedbackResponse {
	status: 'success' | 'error';
	id?: number;
	message?: string;
}

export async function submitFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
	const response = await fetch('/api/feedback', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			chat_id: request.chatId,
			content: request.content
		})
	});

	if (!response.ok) {
		throw new Error('Failed to submit feedback');
	}

	return response.json();
}
