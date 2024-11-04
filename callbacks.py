from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import List, Dict, Any

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    )-> Any:
        """Run when LLM starts running."""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("***********")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any):
        """Run when LLM end running."""
        print(f"***Prompt to LLM was:***\n{response.generations[0][0].text}")
        print("***********")