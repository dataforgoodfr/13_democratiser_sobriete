<script lang="ts">
	import { LogInIcon } from '@lucide/svelte';
</script>

<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
	<div class="w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl">
		<div class="mb-3">
			<h2 class="text-lg font-bold text-gray-900">Welcome to Chat Sufficiency!</h2>
		</div>

		<p class="mb-3 text-sm leading-relaxed text-gray-600">
			This tool allows you to explore the scientific literature on sufficiency policies through a conversational interface.
		</p>

		<div class="mb-5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
			<p class="text-sm leading-relaxed text-amber-800">
				<strong>⚠️ Preview version</strong> — We're still working on the retrieval pipeline
				to build the most precise context possible for the model from the literature.
				Also, as any AI and despite our best efforts, Chat Sufficiency can make mistakes.
				We encourage you to read the cited works.
			</p>
		</div>

		<p class="mb-3 text-sm text-gray-600">Sign in or create a World Sufficiency Lab account to get started:</p>
		<form method="POST" action="/auth/signin/keycloak" target="_top">
			<input type="hidden" name="callbackUrl" value="https://www.thesufficiencylab.org/chat/" />
			<button
				type="submit"
				class="flex w-full items-center justify-center gap-2 rounded-lg bg-gray-900 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-700"
			>
				<LogInIcon size={16} />
				Sign in or register
			</button>
		</form>
	</div>
</div>
