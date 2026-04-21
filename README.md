# Interact-LLM: Inference Experiments
This repository contains code for experiments exploring how large language models (LLMs) perform as cognitive tutors for language learning. All experiments are connected to the `INTERACT-LLM` project. See also the [Research](#-research) section.

>Note: The code is currently *only* intended for internal use and is not production-ready. Currently porting to gemma4-27b.

🔗 [Read about the INTERACT-LLM project (in Danish)](https://cc.au.dk/aktuelt/nyheder/nyhed/artikel/forskere-fra-arts-vil-udvikle-en-chatbot-der-kan-fremme-sprogindlaering),

# 🚀 Overview
The [`src`](/src/) folder contains:
| Folder | Description |
|--------|-------------|
| [`interact_llm`](src/interact_llm/) | Inference engine and terminal-based chatbot |
| [`scripts`](src/scripts/) | Experimental setups using the backend linked to specific publications, including [alignment-drift](/src/scripts/alignment_drift) created for [Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351).|

# 📝 Research 
The `Interact-LLM` repository uses version tags linked to specific papers. Each paper has its own folder in  [/src/scripts](/src/scripts), and the version tag includes the corresponding paper name:
| Paper | `Interact-LLM` Version Suffix| [`scripts`](src/scripts/) folder (Inference code) | Paper Repository (Analysis code) |
|--------|---------------------|------------| ------ |
|[Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351)| `vX.X.X-alignment-drift` | [/src/scripts/alignment_drift](/src/scripts/alignment_drift) | [INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms)|

# 🛠️ Technical Requirements
The code was run on `Python 3.12.3` on both a macOS (`15.3.1`) and Ubuntu system (`24.04`). The project also requires:
| Tool     | Installation                                                                 |
|----------|--------------------------------------------------------------------------------------|
| [make](https://www.gnu.org/software/make/manual/make.html) | Installed via [Homebrew](https://formulae.brew.sh/formula/make)                  |
| [uv](https://docs.astral.sh/uv/)                         | Installed through this project's `makefile` (see [Usage](#usage))                 |

# ⚙️ Usage 

## Setup project 
To install [`uv`](https://docs.astral.sh/uv/) on macOS/Linux and set up a virtual environment with the required Python dependencies, run in the terminal:
```bash
make setup
```

### Add HF Token
If you are interested in running gated models such as `Llama-3.1-8B-Instruct`, you will need to pass a Hugging Face token with read access. 

Create a file called `hf_token.txt` with the token and place it in the [tokens](/tokens) folder. The file will **not** be pushed to `GitHub`.

## Run Chatbot in Terminal App
To experiment with interacting with a chatbot (prompted to act as a Spanish language tutor currently), run in the terminal 
```bash
uv run python -m interact_llm 
```

⚠️ **IMPORTANT**: This is very early development. Functionality is limited.


## Reproduce Experiments 
Refer to the individual READMEs in `scripts` e.g., [alignment-drift](/src/scripts/alignment_drift).
