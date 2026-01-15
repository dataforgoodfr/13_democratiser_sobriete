/// Server-side step to keep backend URL private
/// For now it justs forwards requests to it but we could add auth etc later.

import { CHAT_SUFFICIENCY_API_URL } from '$env/static/private';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		
		const response = await fetch(CHAT_SUFFICIENCY_API_URL, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});

		if (!response.ok) {
			throw error(response.status, 'Failed to fetch from backend');
		}

		// Forward the streaming response
		return new Response(response.body, {
			headers: {
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				'Connection': 'keep-alive'
			}
		});
	} catch (err) {
		console.error('Chat API error:', err);
		throw error(500, 'Internal server error');
	}
};