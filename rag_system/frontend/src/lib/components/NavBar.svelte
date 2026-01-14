<script lang="ts">
	import { SquarePenIcon } from '@lucide/svelte';
	import WSLLogo from '$lib/assets/wsl_logo.svg';
	import { AppBar } from '@skeletonlabs/skeleton-svelte';

	interface Props {
		onReset?: () => void;
	}

	let { onReset }: Props = $props();
</script>

<AppBar class="bg-wsl-yellow p-4">
	<AppBar.Toolbar class="flex w-full items-center gap-6">
		<AppBar.Lead>
			<img class="h-12" alt="World Sufficiency Lab logo" src={WSLLogo} />
		</AppBar.Lead>
		<AppBar.Headline>
			<p class="text-3xl font-bold">Chat Sufficiency</p>
		</AppBar.Headline>
		<AppBar.Trail class="ml-auto">
			<button
				class="flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-white hover:font-semibold"
				onclick={onReset}
			>
				<SquarePenIcon size={20} />
				<span>New chat</span>
			</button>
		</AppBar.Trail>
	</AppBar.Toolbar>
</AppBar>
