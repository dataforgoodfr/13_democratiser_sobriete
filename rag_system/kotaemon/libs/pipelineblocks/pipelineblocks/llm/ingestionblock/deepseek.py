
from pipelineblocks.llm.ingestionblock.base import (
    MetadatasLLMInfBlock,
)
from pydantic import BaseModel

from openai import OpenAI

class DeepSeekMetadatasLLMInference(MetadatasLLMInfBlock):
    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion,
    that produce metadatas inference on doc.

    Attributes:
        llm: The open ai model used for inference.
    """

    llm: OpenAI = OpenAI(
        base_url="https://api.deepseek.com",
        api_key="ollama"
    )

    def run(self, text, doc_type='entire_pdf', inference_type='scientific_advanced',
            existing_metadata: dict | str | None = None) -> BaseModel:


        enriched_prompt = super()._adjust_prompt_according_to_doc_type(
            text, doc_type, inference_type, existing_metadata
        )

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": enriched_prompt}
            ],
            response_format={'type': 'json_object'},
            stream=False
        )

        metadatas = super()._convert_content_to_pydantic_schema(response.choices[0].message.content.strip())

        return metadatas
