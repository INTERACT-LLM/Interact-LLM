add-uv:
	@echo "[INFO:] Installing UV ..."	
	# add mac / linux
	curl -LsSf https://astral.sh/uv/0.6.6/install.sh | sh

install:
	@echo "[INFO:] Installing project ..."
	uv sync

format: 
	@echo "[INFO:] Formatting code with ruff ..."
	uv run ruff format . 						           
	ruff check --select I --fix

check-format: # for later automated formats where pre-commit fails if this check fails
	@echo "[INFO:] Checking formatting ..."
	uv run ruff format . --check						
	uv run ruff check

setup-ollama:
	curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-linux-amd64-rocm.tgz
	sudo tar -C /usr -xzf ollama-linux-amd64-rocm.tgz
	uv add ollama