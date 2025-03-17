"""
MLX wrapper for running quantized mdls
"""

from pathlib import Path
from typing import Optional

from interact_llm.data_models.chat import ChatMessage, ChatHistory

import ollama

class ChatOllama:
    """
    Model wrapper for loading and using a Huggingface model through MLX
    """

    def __init__(
        self,
        model_id: str,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

        # for generation hyperparams (only set once if params passed to generate)
        self.sampling_params = sampling_params
        self.penalty_params = penalty_params

    def load(self) -> None:
        """
        Lazy-loading (loads model and tokenizer if not already loaded)
        """
        ollama.pull(self.model_id)

    def format_params(self):
        if self.sampling_params:
            # normalise "temp" to "temperature" (ensures you can pass temp to the model as this is how MLX/HF defines it)
            if "temp" in self.sampling_params:
                self.sampling_params["temperature"] = self.sampling_params.pop("temp")
            
            kwargs = self.sampling_params
        else:
            kwargs = {}

        if self.penalty_params:
            # normalise repetition penalty
            if "repetition_penalty" in self.penalty_params:
                self.penalty_params["repeat_penalty"] = self.penalty_params.pop("repetition_penalty")
           
            kwargs.update(self.penalty_params)

        return kwargs

    def generate(self, chat: ChatHistory, max_new_tokens: int = -1):
        # format params
        params = self.format_params()
        params["num_predict"] = max_new_tokens

        # chat (decoded output)
        response = ollama.chat(model=self.model_id, 
                               messages= [msg.model_dump() for msg in chat.messages], 
                               options=params)
        
        # formatting
        chat_message = ChatMessage(role="assistant", content=response['message']['content'])

        return chat_message
