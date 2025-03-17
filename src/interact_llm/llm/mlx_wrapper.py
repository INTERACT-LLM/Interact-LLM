"""
MLX wrapper for running quantized mdls
"""

from pathlib import Path
from typing import Optional

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from interact_llm.data_models.chat import ChatMessage


class ChatMLX:
    """
    Model wrapper for loading and using a Huggingface model through MLX
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.device_map = device_map
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None

        # for generation hyperparams (only set once if params passed to generate)
        self.sampler = make_sampler(**sampling_params) if sampling_params else None
        self.logits_processor = (
            make_logits_processors(**penalty_params) if penalty_params else None
        )

    def load(self) -> None:
        """
        Lazy-loading (loads model and tokenizer if not already loaded)
        """
        if self.tokenizer is None or self.model is None:
            self.model, self.tokenizer = load(self.model_id)

            if self.device:
                self.model.to(self.device)

    def generate(self, chat: list, max_new_tokens: int = 3000):
        prompt = self.tokenizer.apply_chat_template(  # nb see https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-1M-4bit
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # chat (decoded output)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=max_new_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processor,
        )

        # formatting
        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message
