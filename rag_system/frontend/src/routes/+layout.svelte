<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Chat Sufficiency</title>
	<meta name="description" content="Chat Sufficiency - Ask about sufficiency policies and research" />
</svelte:head>
{@render children()}
