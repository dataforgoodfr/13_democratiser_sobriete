<script lang="ts">
	// Sign-in page opened in the whole window (target="_top") from the Chat shown in a frame
	// on https://www.thesufficiencylab.org/chat/. The form is sent from this page, same origin as the Chat,
	// so SvelteKit's cross-site check accepts it on every browser.
	import { onMount } from 'svelte';
	import { page } from '$app/state';

	const DEFAUT = 'https://www.thesufficiencylab.org/chat/';
	const demande = page.url.searchParams.get('callbackUrl') ?? '';
	const callbackUrl =
		demande.startsWith('https://www.thesufficiencylab.org/') || (demande.startsWith('/') && !demande.startsWith('//'))
			? demande
			: DEFAUT;

	let formulaire: HTMLFormElement;
	onMount(() => formulaire.requestSubmit());
</script>

<div class="flex min-h-screen items-center justify-center p-6">
	<form bind:this={formulaire} method="POST" action="/auth/signin/keycloak" class="text-center">
		<input type="hidden" name="callbackUrl" value={callbackUrl} />
		<p class="mb-4 text-sm text-gray-600">Redirecting to the World Sufficiency Lab sign-in page…</p>
		<button
			type="submit"
			class="rounded-lg bg-gray-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-gray-700"
		>
			Sign in or register
		</button>
	</form>
</div>
