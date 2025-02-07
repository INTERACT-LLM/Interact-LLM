"""
Chat Model
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import ChatMessage


class ChatHF:
    """
    Model wrapper for loading and using a HuggingFace causal language model
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.device_map = device_map
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """
        Lazy-loading (loads model and tokenizer if not already loaded)
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device_map if self.device_map else None, 
                cache_dir=self.cache_dir
            )
            if self.device:
                self.model.to(self.device)

    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 200):
        ds = datetime.today().strftime("%Y-%m-%d")

        formatted_prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, date_string=ds
        )

        # tokenized inputs and outputs
        token_inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        )

        # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(self.model_id, add_prefix_space=True)

        def get_tokens(word):
            return tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        sequence_bias = [[[get_tokens("rojo")[0]], 8.0, [[get_tokens("manzana")[0]], 6.0]]]

        token_outputs = self.model.generate(
            input_ids=token_inputs.to(self.model.device), max_new_tokens=max_new_tokens, 
            num_beams=4, sequence_bias=sequence_bias,
            repetition_penalty=4.0
        )

        # chat (decoded output)
        response = self.tokenizer.decode((token_outputs[:, token_inputs.shape[1] :])[0])

        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message