import { redirect, type Handle } from '@sveltejs/kit';
import { sequence } from '@sveltejs/kit/hooks';
import { authenticationHandle } from './auth';

const publicPaths = ['/auth', '/'];
const publicPrefixes = ['/_app', '/favicon', '/robots.txt'];

const authorizationHandle: Handle = async ({ event, resolve }) => {
	const pathname = event.url.pathname;

	const isPublicPath =
		publicPaths.some((path) => pathname === path || pathname.startsWith(`${path}/`)) ||
		publicPrefixes.some((prefix) => pathname.startsWith(prefix));

	if (isPublicPath) {
		return resolve(event);
	}

	const session = await event.locals.auth();

	if (!session?.user) {
		if (pathname.startsWith('/api/')) {
			return new Response(JSON.stringify({ error: 'Unauthorized' }), {
				status: 401,
				headers: { 'Content-Type': 'application/json' }
			});
		}

		const callbackUrl = encodeURIComponent(event.url.pathname + event.url.search);
		throw redirect(303, `/auth/signin/keycloak?callbackUrl=${callbackUrl}`);
	}

	return resolve(event);
};

export const handle: Handle = sequence(authenticationHandle, authorizationHandle);
