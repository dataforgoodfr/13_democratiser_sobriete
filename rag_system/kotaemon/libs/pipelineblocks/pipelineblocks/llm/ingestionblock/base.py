from typing import AsyncGenerator, Iterator, Any

from kotaemon.base import BaseComponent, Document
from pydantic import BaseModel
from pipelineblocks.llm.prompts.scientific_paper import scientific_basic_prompt

class BaseLLMIngestionBlock(BaseComponent):

    def stream(self, *args, **kwargs) -> Iterator[Document] | None :
        raise NotImplementedError

    def astream(self, *args, **kwargs) -> AsyncGenerator[Document, None] | None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> Document | list[Document] | Iterator[Document] | None | Any:
        return NotImplementedError


class MetadatasLLMInfBlock(BaseLLMIngestionBlock):
        
    taxonomy : BaseModel

    def _invoke_json_schema_from_taxo(self):

        return self.taxonomy.model_json_schema()

    def _convert_content_to_pydantic_schema(self, content) -> BaseModel:
            
            return self.taxonomy.model_validate_json(content)
    
    def _adjust_prompt_according_to_doc_type(self, text, doc_type, inference_type) -> str:

        if inference_type == 'scientific' and doc_type == 'entire_doc':
            # First combination example
            enriched_prompt = scientific_basic_prompt(text)
            
        elif inference_type == 'scientific' and doc_type == 'chunk':
            # Other combination Example
            raise NotImplementedError(f"The {inference_type} inference type is not implemented for this doc_type : {doc_type} ")
        
        else:
            raise NotImplementedError(f"The {inference_type} inference type is not implemented for this doc_type : {doc_type} ")
        
        return enriched_prompt
    

    def run(self, *args, **kwargs) -> BaseModel:
        return NotImplementedError
    
# TODO --- Exemple
class SummarizationLLMInfBlock(BaseLLMIngestionBlock):

    def __init__(self, *args, **kwargs):

        pass

    def run(self, *args, **kwargs):
        return NotImplementedError