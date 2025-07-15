import time

from pipelineblocks.llm.ingestionblock.base import (
    CustomPromptLLMInfBlock,
    MetadatasLLMInfBlock,
)
from pydantic import BaseModel

from kotaemon.base.schema import HumanMessage
from kotaemon.llms.chats.langchain_based import LCChatMistral

# All the OpenAI LLM Inference blocks are mainly used for
# Ollama models (local deployment) according to the Kotaemon logic

# Don't hesitate to create a new file for other packaging logic like Ollama (without Kotaemon)...


class LangChainMetadatasLLMInference(MetadatasLLMInfBlock):

    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion,
    that produce metadatas inference on doc.

    Attributes:
        llm: The open ai model used for inference.
    """

    llm: LCChatMistral = LCChatMistral.withx(
        model="open-mistral-nemo",
        mistral_api_key="*************************",
        temperature=0,
    )

    def run(
        self, text, doc_type="entire_pdf", inference_type="scientific"
    ) -> BaseModel:

        self.llm.max_tokens = 1024

        enriched_prompt = super()._adjust_prompt_according_to_doc_type(
            text, doc_type, inference_type
        )

        if self.language != "English":
            system_message = self._build_a_system_message_to_force_language(
                language=self.language
            )
            messages = [system_message, HumanMessage(content=enriched_prompt)]
        else:
            messages = HumanMessage(content=enriched_prompt)

        struct_llm = self.llm.with_structured_output(self.taxonomy)

        input_ = self.llm.prepare_message(messages)

        response = struct_llm.invoke(input_)

        # First retry strategy
        if not response or not isinstance(response, dict):
            for _ in range(3):
                time.sleep(1)
                response = struct_llm.invoke(input_)
                if response or isinstance(response, dict):
                    break
        # Second retry strategy
        if not response or not isinstance(response, dict):
            struct_llm = self.llm.with_structured_output(
                self.taxonomy, include_raw=True
            )
            response = struct_llm.invoke(input_)
            response = str(response["raw"].content)
            if response != "" and response != " ":
                response = super()._convert_content_to_pydantic_schema(
                    response, pydantic_schema=self.taxonomy
                )

        # TODO validate model
        """
        if type(response) is str:

            metadatas = super()._convert_content_to_pydantic_schema(response, mode='json')

        else: # dict output

            metadatas = super()._convert_content_to_pydantic_schema(response, mode='dict')"""

        return response


class LangChainCustomPromptLLMInference(CustomPromptLLMInfBlock):

    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion,
    that produces inference according to a custom prompt.
    This prompts should finish with 'This is the text :', 'This is the doc: ' or 'This is the context : '

    Attributes:
        llm: The open ai model used for inference.
    """

    llm: LCChatMistral = LCChatMistral.withx(
        model="open-mistral-nemo", mistral_api_key="*************************"
    )

    def run(
        self,
        messages,
        temperature: float | None = None,
        language: str = "English",
        pydantic_schema: BaseModel | None = None,
        extra_max_retries: int = 10,
        max_token: int = 1024,
    ) -> BaseModel | str:

        # Overwrite temperature (id needed)
        if temperature is not None:
            self.llm.temperature = temperature

        self.llm.max_tokens = max_token

        if language != "English":
            system_message = self._build_a_system_message_to_force_language(
                language=language
            )
            total_messages = [system_message, *messages]
        else:
            total_messages = messages

        if pydantic_schema is not None:

            struct_llm = self.llm.with_structured_output(pydantic_schema)

            input_ = self.llm.prepare_message(total_messages)

            response = struct_llm.invoke(input_)

            # First retry strategy
            if not response or not isinstance(response, dict):
                for _ in range(extra_max_retries):
                    time.sleep(1)
                    response = struct_llm.invoke(input_)
                    if response or isinstance(response, dict):
                        break
            # Second retry strategy
            if not response or not isinstance(response, dict):
                struct_llm = self.llm.with_structured_output(
                    pydantic_schema, include_raw=True
                )
                response = struct_llm.invoke(input_)
                response = str(response["raw"].content)
                if response != "" and response != " ":
                    response = super()._convert_content_to_pydantic_schema(
                        response, pydantic_schema=pydantic_schema
                    )
                else:
                    raise RuntimeError("Empty response")

            return response

        else:

            response = self.llm.invoke(messages=total_messages)

            return response.content
