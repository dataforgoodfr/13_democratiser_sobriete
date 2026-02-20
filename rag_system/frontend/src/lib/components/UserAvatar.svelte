<script lang="ts">
	interface Props {
		userName: string;
		/** 'dark'  → black avatar, white dropdown (for yellow AppBar)
		 *  'light' → white avatar with black border, themed dropdown (for minimal header) */
		variant?: 'light' | 'dark';
	}

	let { userName, variant = 'light' }: Props = $props();

	let showMenu = $state(false);

	function getInitials(name: string): string {
		return name
			.trim()
			.split(/\s+/)
			.map((part) => part[0]?.toUpperCase() ?? '')
			.slice(0, 2)
			.join('');
	}

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (!target.closest('[data-user-avatar]')) {
			showMenu = false;
		}
	}
</script>

<svelte:window onclick={handleClickOutside} />

<div class="relative" data-user-avatar>
	<button
		class={[
			'flex h-9 w-9 items-center justify-center rounded-full text-sm font-semibold hover:opacity-80',
			variant === 'dark'
				? 'bg-black text-white'
				: 'bg-background border border-black text-black'
		].join(' ')}
		onclick={(e) => { e.stopPropagation(); showMenu = !showMenu; }}
		title={userName}
	>
		{getInitials(userName)}
	</button>

	{#if showMenu}
		<div
			class={[
				'absolute right-0 top-11 z-50 min-w-40 rounded-lg border shadow-lg',
			].join(' ')}
		>
			<div
				class={[
					'border-b px-4 py-2 text-sm text-muted-foreground',
					variant === 'dark' ? 'border-gray-200 text-black' : ''
				].join(' ')}
			>
				{userName}
			</div>
			<form method="POST" action="/?/signOut" class="p-1">
				<button
					class={[
						'w-full rounded px-3 py-1.5 text-left text-sm font-semibold hover:bg-gray-100',
					].join(' ')}
					type="submit"
				>
					Sign out
				</button>
			</form>
		</div>
	{/if}
</div>
