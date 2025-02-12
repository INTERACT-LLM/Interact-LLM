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

        # sequence biasing
        self.tokenizer_with_prefix_space = None # tokenizer for sequence biasing 
        self.sequence_bias: list[list[list[str], float]] = []

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
                

    def tokenize_for_sequence_bias(self, word:str) -> int: 
        """special tokenisation for sequencebiasing"""
        # load only the first time
        if self.tokenizer_with_prefix_space is None:
            self.tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(self.model_id, add_prefix_space=True)

        token = self.tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0][0]

        return token

    def _create_sequence_bias(self, words:list[str], bias:list[float] | float) -> None:
        """
        Create sequence bias for biased generation.

        Args
            words: list of words for biased generation to be tokenised
            bias: list of biases correspondning to each word in 'words' OR a single bias value (i.e., uniform bias to each word in 'words')
        """
        if not self.sequence_bias:
            if isinstance(bias, list):
                if len(words) != len(bias):
                    raise ValueError("[ERROR:] Lists 'bias' and 'words' are not the same length. Pass two lists of same length or pass 'bias' as a single float value")
            else: 
                bias = [bias] * len(words) # (give same bias to each word if single float bias)
                
            for word, bias_value in zip(words, bias):
                 token = self.tokenize_for_sequence_bias(word)
                 self.sequence_bias.append([[token], bias_value]) # format that HF wants

    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 200, words:list[str]=None, bias:list[float] | float = None): # rename from generate to chat
        ds = datetime.today().strftime("%Y-%m-%d") # pass to fn in future

        formatted_prompt = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True, 
            date_string=ds
        )

        # tokenized inputs and outputs
        input_ids = self.tokenizer.encode(
            formatted_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        )

        if words and bias: 
            self._create_sequence_bias(words, bias)
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.model.device), 
                max_new_tokens=max_new_tokens, 
                sequence_bias=self.sequence_bias,
                do_sample=True,
            )

        else: 
            output_ids = self.model.generate(
            input_ids=input_ids.to(self.model.device), 
            max_new_tokens=max_new_tokens
        )

        # chat (decoded output)
        response = self.tokenizer.decode((output_ids[:, input_ids.shape[1] :])[0])

        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message