import { SvelteKitAuth } from '@auth/sveltekit';
import Keycloak from '@auth/sveltekit/providers/keycloak';
import { TOPSITE_URL } from '$env/static/private';

export const { handle: authenticationHandle, signIn, signOut } = SvelteKitAuth({
	providers: [Keycloak],
	trustHost: true,
	callbacks: {
		redirect({ url, baseUrl }) {
			if (url.startsWith(TOPSITE_URL)) {
				return url;
			}
			return url.startsWith('/') ? `${baseUrl}${url}` : baseUrl;
		}
	}
});
