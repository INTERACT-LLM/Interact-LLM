from pathlib import Path
from typing import Optional

from interact_llm.data_models.chat import ChatMessage
from llama_cpp import Llama

class ChatLlamacpp:
    """
    Model wrapper for loading and using a Huggingface model through MLX
    """

    def __init__(
        self,
        model_id: str,
        filename: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        context_length: Optional[int] = None,
        n_gpu_layers: Optional[int] = None, # -1 for gpu 
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.filename = filename
        self.context_length = context_length
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self.sampling_params = sampling_params
        self.penalty_params = penalty_params

    def load(self) -> None:
        """
        Lazy-loading (loads model and tokenizer if not already loaded)
        """
        if self.model is None:
            self.model = Llama.from_pretrained( # requires huggingface-hub
                repo_id = self.model_id, 
                filename = self.filename,
                verbose = False,
                n_ctx = self.context_length
            )

    def format_params(self):
        if self.sampling_params:
            # normalize "temp" to "temperature" (ensures you can pass temp to the model as this is how MLX/HF defines it)
            if "temp" in self.sampling_params:
                self.sampling_params["temperature"] = self.sampling_params.pop("temp")
            
            kwargs = self.sampling_params
        else:
            kwargs = {}

        if self.penalty_params:
            kwargs.update(self.penalty_params)

        return kwargs

    def generate(self, chat: list, max_new_tokens: int = 200):
        kwargs = self.format_params()

        # chat (decoded output)
        response = self.model.create_chat_completion(
            messages = chat,
            max_tokens = max_new_tokens,
            **kwargs
        )

        # formatting
        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message
