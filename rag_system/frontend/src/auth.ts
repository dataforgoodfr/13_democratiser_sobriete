import { SvelteKitAuth } from '@auth/sveltekit';
import Keycloak from '@auth/sveltekit/providers/keycloak';
import { TOPSITE_URL } from '$env/static/private';
import { env } from '$env/dynamic/private';

// Shared sign-out across the World Sufficiency Lab apps (same Keycloak realm as the Atlas and Zero Carbon):
// every 5 minutes at most, the Chat checks that the Keycloak session is still open by refreshing the token.
// If the person signed out elsewhere (Atlas, Community…) or the session expired, Keycloak refuses the refresh
// and the Chat session ends too. If Keycloak cannot be reached, nobody is signed out.
const CHECK_EVERY_MS = 5 * 60 * 1000;

async function keycloakSessionStillOpen(refreshToken: string): Promise<{ open: boolean; refreshToken?: string }> {
	try {
		const r = await fetch(`${env.AUTH_KEYCLOAK_ISSUER}/protocol/openid-connect/token`, {
			method: 'POST',
			headers: { 'content-type': 'application/x-www-form-urlencoded' },
			body: new URLSearchParams({
				grant_type: 'refresh_token',
				client_id: env.AUTH_KEYCLOAK_ID ?? '',
				client_secret: env.AUTH_KEYCLOAK_SECRET ?? '',
				refresh_token: refreshToken
			})
		});
		if (r.status === 400 || r.status === 401) return { open: false }; // invalid_grant: session closed in Keycloak
		if (!r.ok) return { open: true };
		const body = await r.json();
		return { open: true, refreshToken: body.refresh_token ?? refreshToken };
	} catch {
		return { open: true };
	}
}

export const { handle: authenticationHandle, signIn, signOut } = SvelteKitAuth({
	providers: [Keycloak],
	trustHost: true,
	callbacks: {
		redirect({ url, baseUrl }) {
			if (url.startsWith(TOPSITE_URL)) {
				return url;
			}
			return url.startsWith('/') ? `${baseUrl}${url}` : baseUrl;
		},
		async jwt({ token, account }) {
			if (account) {
				return { ...token, refreshToken: account.refresh_token, checkedAt: Date.now() };
			}
			const refreshToken = token.refreshToken as string | undefined;
			const checkedAt = (token.checkedAt as number | undefined) ?? 0;
			if (!refreshToken || Date.now() - checkedAt < CHECK_EVERY_MS) {
				return token;
			}
			const check = await keycloakSessionStillOpen(refreshToken);
			if (!check.open) {
				return null; // ends the Chat session
			}
			return { ...token, refreshToken: check.refreshToken, checkedAt: Date.now() };
		}
	}
});
