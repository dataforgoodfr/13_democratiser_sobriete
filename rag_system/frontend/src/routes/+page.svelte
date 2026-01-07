<script lang="ts">
  import {
    Conversation,
    ConversationContent,
    ConversationEmptyState,
    ConversationScrollButton,
  } from "$lib/components/ai-elements/conversation";
  import { Message, MessageContent } from "$lib/components/ai-elements/message";
  import {
    PromptInput,
    PromptInputTextarea,
    PromptInputSubmit,
  } from "$lib/components/ai-elements/prompt-input";
  import { Response } from "$lib/components/ai-elements/response";
  import { MessageSquare } from "lucide-svelte";

  interface ChatMessage {
    role: "user" | "assistant";
    content: string;
  }

  const API_URL = "http://localhost:8000/api/chat";

  let input = $state("");
  let messages: ChatMessage[] = $state([]);
  let status: "idle" | "streaming" = $state("idle");

  async function handleSubmit(message: { text?: string }, event: SubmitEvent) {
    event.preventDefault();
    if (!input.trim() || status === "streaming") return;

    const userMessage: ChatMessage = { role: "user", content: input };
    messages = [...messages, userMessage];
    input = "";
    status = "streaming";

    // Add placeholder for assistant response
    const assistantMessage: ChatMessage = { role: "assistant", content: "" };
    messages = [...messages, assistantMessage];

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
      });

      if (!response.ok) throw new Error("Failed to fetch response");

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No reader available");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") break;
            
            // Update the last message (assistant's response)
            messages = messages.map((msg, i) =>
              i === messages.length - 1
                ? { ...msg, content: msg.content + data }
                : msg
            );
          }
        }
      }
    } catch (error) {
      console.error("Error:", error);
      // Update last message to show error
      messages = messages.map((msg, i) =>
        i === messages.length - 1
          ? { ...msg, content: "Error: Failed to get response" }
          : msg
      );
    } finally {
      status = "idle";
    }
  }
</script>

<div class="flex min-h-screen items-center justify-center bg-background p-4">
  <div class="max-w-4xl w-full h-[600px] rounded-lg border bg-card p-6">
    <div class="flex flex-col h-full">
    <Conversation>
      <ConversationContent>
        {#if messages.length === 0}
          <ConversationEmptyState
            title="Start a conversation"
            description="Type a message below to begin chatting"
          >
            {#snippet icon()}
              <MessageSquare class="size-12" />
            {/snippet}
          </ConversationEmptyState>
        {:else}
          {#each messages as message, messageIndex (messageIndex)}
            <Message from={message.role}>
              <MessageContent>
                <Response content={message.content} />
              </MessageContent>
            </Message>
          {/each}
        {/if}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>

    <PromptInput onSubmit={handleSubmit} class="mt-4 w-full max-w-2xl mx-auto relative">
      <PromptInputTextarea
        bind:value={input}
        placeholder="Say something..."
        class="pr-12"
      />
      <PromptInputSubmit
        status={status}
        disabled={!input.trim()}
        class="absolute bottom-1 right-1"
      />
    </PromptInput>
    </div>
  </div>
</div>