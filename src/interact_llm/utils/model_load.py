"""
Utils for model loading either with a HF or MLX backend
"""

from pathlib import Path
from typing import Literal, Optional

import toml

from interact_llm.llm.hf_wrapper import ChatHF
from interact_llm.llm.mlx_wrapper import ChatMLX


def get_model_id(
    models_config_path: Path, model_name: str, backend: Literal["mlx", "hf"] = "mlx"
) -> str:
    """
    Reads models from a TOML file and returns the correct model ID
    based on the specified backend ('mlx' or 'hf') and an input model ID.

    models_config_path: Path to the models.toml file (usually placed in /configs)
    model_name: name defined in toml with corresponding backends
    backend: Either 'mlx' or 'hf'

    returns:
        The corresponding model ID for the selected backend
    """
    if models_config_path.suffix != ".toml":
        raise ValueError(f"The file at '{models_config_path}' is not a TOML file.")

    if backend not in {"mlx", "hf"}:
        raise ValueError("Backend must be 'mlx' or 'hf'")

    models = toml.load(models_config_path)["models"]

    for model in models:
        if model_name in model.values():
            if backend in model:
                return model[backend]
            raise ValueError(
                f"Model '{model_name}' exists but has no backend '{backend}' entry."
            )

    model_names = [model["name"] for model in models]
    raise ValueError(
        f"No model defined for '{model_name}. Add it to '{models_config_path.name}' or choose between defined models: {model_names}'"
    )


def login_hf_token(token_path: Path = Path(__file__).parents[3] / "tokens" / "hf_token.txt") -> None:
    """
    Load HF token from "tokens" folder and login.
    """
    from huggingface_hub import login

    try:
        # get token from txt
        with open(token_path) as f:
            hf_token = f.read().strip()

        login(hf_token)
        print("Logged in to Hugging Face successfully.")
    
    except Exception as e:
        print(f"Error during Hugging Face login: {e}")
        raise  # Re-raise the exception after printing

def load_model_backend(
    models_config_path: Path,
    model_name: str,
    backend: Literal["mlx", "hf"] = "mlx",
    token_path: Path = Path(__file__).parents[3] / "tokens" / "hf_token.txt",
    cache_dir: Optional[Path] = None,
    **model_kwargs,
) -> ChatHF | ChatMLX:
    """
    Loads a model based on the specified backend ("mlx" or "hf"). Will try to login to HF 

    Args:
        models_config_path: Path to the models configuration.
        model_name: The name of the model to load.
        backend: The backend to use for loading the model, default is "mlx".
        **model_kwargs: Additional keyword arguments passed to the model's initialization 
            (e.g., sampling params, see documentation for ChatHF or ChatMLX)

    Returns:
        ChatHF | ChatMLX: The loaded model object.
    """
    model_id = get_model_id(
        models_config_path=models_config_path, model_name=model_name, backend=backend
    )

    # instantiate model based on backend
    if backend == "mlx":
        model = ChatMLX(model_id=model_id, **model_kwargs)
    elif backend == "hf":
        model = ChatHF(model_id=model_id, cache_dir=cache_dir, **model_kwargs)

    try: 
        model.load()
        print(f"Model {model_name} loaded successfully using {backend} backend (model_id = {model_id})")

    except OSError as e:
        # If the error is related to gated access (authentication error)
        if '401 Client Error' in str(e):
            print(f"Error loading model {model_name} from {backend} backend: {e}")
            print("Attempting to log in to Hugging Face...")
            login_hf_token(token_path)  
            model.load()
            print(f"Model {model_name} loaded successfully using {backend} backend (model_id = {model_id})")
        else:
            print(f"Unexpected error occurred: {e}")
            raise  #rReraise other unexpected errors

    return model