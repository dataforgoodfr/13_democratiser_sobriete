/// Server-side step to keep backend URL private
/// For now it just forwards requests to it but we could add auth etc later.

import { CHAT_SUFFICIENCY_API_URL } from '$env/static/private';
import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();

		const response = await fetch(`${CHAT_SUFFICIENCY_API_URL}/api/feedback`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});

		if (!response.ok) {
			throw error(response.status, 'Failed to submit feedback to backend');
		}

		const data = await response.json();
		return json(data);
	} catch (err) {
		console.error('Feedback API error:', err);
		throw error(500, 'Internal server error');
	}
};
