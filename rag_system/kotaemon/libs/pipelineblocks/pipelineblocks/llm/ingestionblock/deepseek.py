from pipelineblocks.llm.ingestionblock.base import (
    MetadatasLLMInfBlock,
)
from pydantic import BaseModel

import instructor
from taxonomy.paper_taxonomy import PaperTaxonomy

class DeepSeekMetadatasLLMInference(MetadatasLLMInfBlock):
    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion,
    that produce metadatas inference on doc.

    Attributes:
        llm: The open ai model used for inference.
    """



    def run(self, text, doc_type='entire_pdf', inference_type='scientific_advanced',
            existing_metadata: dict | str | None = None) -> BaseModel:
        enriched_prompt = super()._adjust_prompt_according_to_doc_type(
            text, doc_type, inference_type, existing_metadata
        )

        llm = instructor.from_provider(
            "deepseek/deepseek-reasoner",
            async_client=True,
            base_url="https://api.deepseek.com",
        )

        response = llm.chat.completions.create(
            messages=[
                {"role": "user", "content": enriched_prompt}
            ],
            response_model=PaperTaxonomy
        )

        metadatas = super()._convert_content_to_pydantic_schema(response.choices[0].message.content.strip())

        return metadatas
