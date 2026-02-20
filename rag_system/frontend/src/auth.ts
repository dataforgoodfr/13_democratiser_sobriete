import { SvelteKitAuth } from '@auth/sveltekit';
import Keycloak from '@auth/sveltekit/providers/keycloak';

export const { handle: authenticationHandle, signIn, signOut } = SvelteKitAuth({
	providers: [Keycloak],
	trustHost: true
});
