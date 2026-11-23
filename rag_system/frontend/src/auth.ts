import { redirect, type RequestHandler } from '@sveltejs/kit';
import { TOPSITE_URL } from '$env/static/private';
import { env } from '$env/dynamic/private';

// Shared sign-out: ends the Chat session, then the Keycloak session (so the Atlas, Zero Carbon and the other
// World Sufficiency Lab apps sign out too), then comes back to the site. Opened in the whole window
// (target="_top"): Keycloak cannot be shown in a frame. A plain link, not a form, so that it also works
// from the Chat shown in a frame on browsers that send "Origin: null".
export const GET: RequestHandler = ({ cookies }) => {
	for (const { name } of cookies.getAll()) {
		if (/^(__Secure-|__Host-)?authjs\.(session-token|csrf-token|callback-url)/.test(name)) {
			cookies.delete(name, { path: '/' });
		}
	}
	const back = TOPSITE_URL || '/';
	if (!env.AUTH_KEYCLOAK_ISSUER) {
		throw redirect(303, back);
	}
	const params = new URLSearchParams({ client_id: env.AUTH_KEYCLOAK_ID ?? '', post_logout_redirect_uri: back });
	throw redirect(303, `${env.AUTH_KEYCLOAK_ISSUER}/protocol/openid-connect/logout?${params}`);
};
