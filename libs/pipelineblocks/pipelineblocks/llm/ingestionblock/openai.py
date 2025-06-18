from pipelineblocks.llm.ingestionblock.base import (
    CustomPromptLLMInfBlock,
    MetadatasLLMInfBlock,
)
from pydantic import BaseModel

from kotaemon.base.schema import HumanMessage
from kotaemon.llms.chats.openai import ChatOpenAI

# All the OpenAI LLM Inference blocks are mainly used for
# Ollama models (local deployment) according to the Kotaemon logic

# Don't hesitate to create a new file for other packaging logic like Ollama (without Kotaemon)...


class OpenAIMetadatasLLMInference(MetadatasLLMInfBlock):

    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion,
    that produce metadatas inference on doc.

    Attributes:
        llm: The open ai model used for inference.
    """

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url="http://localhost:11434/v1/",
        model="gemma2:2b",
        api_key="ollama",
    )

    def run(
        self, text, doc_type="entire_pdf", inference_type="scientific"
    ) -> BaseModel:

        json_schema = super()._invoke_json_schema_from_taxo()

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

        response = self.llm.invoke(
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": json_schema,
                    "name": "output_schema",
                    "strict": True,
                },
            },
        )

        metadatas = super()._convert_content_to_pydantic_schema(response.content)

        return metadatas


class OpenAICustomPromptLLMInference(CustomPromptLLMInfBlock):

    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion,
    that produces inference according to a custom prompt.
    This prompts should finish with 'This is the text :', 'This is the doc: ' or 'This is the context : '

    Attributes:
        llm: The open ai model used for inference.
    """

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url="http://localhost:11434/v1/",
        model="gemma2:2b",
        api_key="ollama",
    )

    def run(
        self,
        text: str,
        messages,
        temperature: float = 0.3,
        language: str = "English",
        pydantic_schema: BaseModel | None = None,
    ) -> BaseModel | str:

        if language != "English":
            system_message = self._build_a_system_message_to_force_language(
                language=language
            )
            messages = [
                system_message,
                *messages,
                HumanMessage(content=f"\n {text} \n"),
            ]
        else:
            messages = [*messages, HumanMessage(content=f"\n {text} \n")]

        if pydantic_schema is not None:

            json_schema = super()._invoke_json_schema_from_pydantic_schema(
                pydantic_schema=pydantic_schema
            )

            response = self.llm.invoke(
                messages=messages,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": json_schema,
                        "name": "output_schema",
                        "strict": True,
                    },
                },
            )

            response_schema = super()._convert_content_to_pydantic_schema(
                content=response.content, pydantic_schema=pydantic_schema
            )

            return response_schema

        else:

            response = self.llm.invoke(messages=messages, temperature=temperature)

            return response.content


# TODO -- Example with summarization
class OpenAISummarizationLLMInference(MetadatasLLMInfBlock):
    # TODO -- Example with summarization
    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion, with some inference.
    Attributes:
        model: The open ai model used for inference.
    """

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url="http://localhost:11434/v1/",
        model="gemma2:2b",
        api_key="ollama",
    )

    def run(
        self, text, doc_type="entire_pdf", inference_type="scientific"
    ) -> BaseModel:
        # TODO -- Example with summarization
        json_schema = super()._invoke_json_schema_from_taxo()

        enriched_prompt = super()._adjust_prompt_according_to_doc_type(
            text, doc_type, inference_type
        )

        response = self.llm.invoke(
            messages=HumanMessage(content=enriched_prompt),
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": json_schema,
                    "name": "output_schema",
                    "strict": True,
                },
            },
        )

        metadatas = super()._convert_content_to_pydantic_schema(response.content)
        # TODO -- Example with summarization
        return metadatas
