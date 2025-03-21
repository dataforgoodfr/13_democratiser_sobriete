

from pipelineblocks.llm.ingestionblock.base import MetadatasLLMInfBlock
from kotaemon.llms.chats.openai import ChatOpenAI
from pydantic import BaseModel
from kotaemon.base.schema import HumanMessage

# All the OpenAI LLM Inference blocks are mainly used for 
# Ollama models (local deployment) according to the Kotaemon logic

# Don't hesitate to create a new file for other packaging logic like Ollama (without Kotaemon)...

class OpenAIMetadatasLLMInference(MetadatasLLMInfBlock):
        
    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion, with some inference.
    Attributes:
        model: The open ai model used for inference.
    """

    llm : ChatOpenAI = ChatOpenAI.withx(
            base_url="http://localhost:11434/v1/",
            model="gemma2:2b",
            api_key="ollama",
            )

    def run(self, text,  doc_type  = 'entire_pdf', inference_type = 'scientific') -> BaseModel:

        json_schema = super()._invoke_json_schema_from_taxo()

        enriched_prompt = super()._adjust_prompt_according_to_doc_type(text, doc_type, inference_type)
        
        response = self.llm.invoke(
                messages= HumanMessage(content=enriched_prompt),
                temperature=0,
                response_format={"type":"json_schema",
                                "json_schema": {"schema":json_schema,
                                                "name":"output_schema",
                                                "strict": True}
                                }
                            )
        
        metadatas = super()._convert_content_to_pydantic_schema(response.content)

        return metadatas


# TODO -- Example with summarization

class OpenAISummarizationLLMInference(MetadatasLLMInfBlock):
        
    """
    A special OpenAI model (included 'Ollama model' with Kotaemon style) block ingestion, with some inference.
    Attributes:
        model: The open ai model used for inference.
    """

    llm : ChatOpenAI = ChatOpenAI.withx(
            base_url="http://localhost:11434/v1/",
            model="gemma2:2b",
            api_key="ollama",
            )

    def run(self, text,  doc_type  = 'entire_pdf', inference_type = 'scientific') -> BaseModel:

        json_schema = super()._invoke_json_schema_from_taxo()

        enriched_prompt = super()._adjust_prompt_according_to_doc_type(text, doc_type, inference_type)
        
        response = self.llm.invoke(
                messages= HumanMessage(content=enriched_prompt),
                temperature=0,
                response_format={"type":"json_schema",
                                "json_schema": {"schema":json_schema,
                                                "name":"output_schema",
                                                "strict": True}
                                }
                            )
        
        metadatas = super()._convert_content_to_pydantic_schema(response.content)

        return metadatas