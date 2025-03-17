"""
Simulate two chat LLMs talking to each other

Model tested currently:
- mlx-community/Qwen2.5-7B-Instruct-1M-4bit
- mlx-community/meta-Llama-3.1-8B-Instruct-4bit

"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from interact_llm.data_models.chat import ChatHistory, ChatMessage
from interact_llm.data_models.prompt import SystemPrompt, load_prompt_by_id
from interact_llm.llm.hf_wrapper import ChatHF
from interact_llm.llm.mlx_wrapper import ChatMLX
from interact_llm.utils.model_load import load_model_backend
from scripts.detect_lang import _detect_lang

DEFAULT_PROMPT_VERSION = 3.0


def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--prompt_id", help="id of prompt in toml", type=str, default="A1"
    )
    parser.add_argument(
        "--prompt_version",
        help="version of prompt toml file in configs/prompts/",
        type=float,
        default=DEFAULT_PROMPT_VERSION,
    )

    parser.add_argument(
        "--model_name",
        help="model name as specified in configs/models.toml",
        type=str,
        default="qwen2.5:7b",
    )

    parser.add_argument(
        "--backend",
        help="whether to run a quantized model with MLX or a model with HF (transformers)",
        type=str,
        default="hf",
    )

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args


def simulate_conversation(
    model: ChatMLX | ChatHF, n_total_rounds: int = 9, tutor_system_prompt=SystemPrompt
) -> ChatHistory:
    """
    Simulate an LLM conversation

    Note that we are interested in the tutor only, but each has their own history in which they are the assistant, responding to a user.
    """

    # define histories
    student_history = ChatHistory(
        messages=[
            ChatMessage(
                role="system",
                content="You are a student learning Spanish, responding to a teacher who is facilitating a natural dialogue with you.",
            )
        ]
    )

    tutor_history = ChatHistory(
        messages=[
            ChatMessage(
                role=tutor_system_prompt.role, content=tutor_system_prompt.content
            ),
            ChatMessage(
                role="user", content="Hola"
            ),  # pre-fixed what the tutor LLM receives
        ]
    )

    for _ in tqdm(range(n_total_rounds)):
        # tutor in assistant role responds to user (first time to the pre-fixed "hola")
        max_retries = 5
        tutor_message = None

        for attempt in range(max_retries):
            tutor_message = model.generate(tutor_history)
            if not _detect_lang(tutor_message.content):  # If no English is detected, proceed
                break
            print(f"[WARNING]: Tutor response contains English (attempt {attempt + 1}/{max_retries}). Regenerating...")

        else: 
            print("[ERROR]: Tutor failed to generate a fully Spanish response after max retries. Returning None...")
            return None 

        tutor_history.messages.append(tutor_message)

        # student receives tutor response as a user message
        student_history.messages.append(
            ChatMessage(role="user", content=tutor_message.content)
        )

        # student in assistant role responds to user, append to teacher chat history
        student_message = model.generate(student_history)
        student_history.messages.append(student_message)

        # tutor receives student response as a user message
        tutor_history.messages.append(
            ChatMessage(role="user", content=student_message.content)
        )

    return tutor_history


def main():
    args = input_parse()

    n_runs = 10

    for n in range(n_runs):
        print(f"[INFO]: Running simulation run {n + 1} out of {n_runs}")

        # MODEL LOADING
        sampling_params = {
            "temp": 0.8,
            "top_p": 0.95,
            "min_p": 0.05,
            "top_k": 40,
        }  # default params on LM studio and llama.cpp (https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/types.py#L25)
        penalty_params = {"repetition_penalty": 1.1}

        cache_dir = Path(__file__).parents[3] / "models"
        models_config_file = Path(__file__).parents[2] / "configs" / "models.toml"

        model = load_model_backend(
            models_config_path=models_config_file,
            model_name=args.model_name,
            backend=args.backend,
            token_path=Path(__file__).parents[2] / "tokens" / "hf_token.txt",
            cache_dir=cache_dir if args.backend == "hf" else None,
            sampling_params=sampling_params,
            penalty_params=penalty_params
        )

        # PROMPT FORMATTING
        prompt_version = args.prompt_version
        prompt_id = args.prompt_id
        prompt_file = (
            Path(__file__).parents[2]
            / "configs"
            / "prompts"
            / f"v{str(prompt_version)}.toml"
        )

        print(
            f"[INFO]: Formatting prompts using toml file version {prompt_version} and prompt id {prompt_id}"
        )

        system_prompt = load_prompt_by_id(
            toml_path=prompt_file, prompt_id=prompt_id, system_prompt=True
        )

        # simulate
        tutor_history = simulate_conversation(
            model=model, n_total_rounds=9, tutor_system_prompt=system_prompt
        )

        if tutor_history is None:
            print(f"[INFO]: Skipping run {n + 1}")
            continue  # skip this run and continue to the next one

        # save chat
        chat_json = json.dumps(
            [msg.model_dump() for msg in tutor_history.messages],
            indent=3,
            ensure_ascii=False,
        )

        save_dir = (
            Path(__file__).parents[3]
            / "simulated_data"
            / model.model_id.replace("/", "--")
            / f"v{str(prompt_version)}"
            / prompt_id
        )

        save_dir.mkdir(exist_ok=True, parents=True)

        save_file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(save_dir / f"{save_file_name}.json", "w") as outfile:
            outfile.write(chat_json)

        # remove from mem 
        del model


if __name__ == "__main__":
    main()
