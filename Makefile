ifeq (,$(wildcard .env))
$(error .env file is missing. Please create one based on .env.example)
endif

include .env

CHECK_DIRS := src/

# --- Ruff ---

format-fix:
	uv run ruff format $(CHECK_DIRS) 
	uv run ruff check --select I --fix

lint-fix:
	uv run ruff check --fix $(CHECK_DIRS) 

format-check:
	uv run ruff format --check $(CHECK_DIRS) 
	uv run ruff check -e $(CHECK_DIRS)
	uv run ruff check --select I -e

lint-check:
	uv run ruff check $(CHECK_DIRS)

# --- Docker ---

build-faster-whisper-image:
	docker build -f Dockerfile.faster_whisper \
		--build-arg FASTER_WHISPER_MODEL=$(FASTER_WHISPER__MODEL) \
		-t theneuralmaze/faster-whisper-server:latest .

push-faster-whisper-image:
	docker push theneuralmaze/faster-whisper-server:latest

build-orpheus-image:
	docker build -f Dockerfile.orpheus \
		-t theneuralmaze/orpheus-server:latest .

push-orpheus-image:
	docker push theneuralmaze/orpheus-server:latest

# --- RunPod ---

create-faster-whisper-pod:
	uv run python scripts/runpod/create_faster_whisper_pod.py

create-orpheus-pod:
	uv run python scripts/runpod/create_orpheus_pod.py


# --- Run Gradio ---

start-gradio-application:
	uv run python scripts/run_gradio_application.py

# --- Application Local Deployment ---

start-call-center:
	docker compose up --build -d

stop-call-center:
	docker compose down

delete-call-center:
	docker compose down -v

start-ngrok-tunnel:
	@echo "Starting ngrok tunnel on port 8000..."
	@echo "Remember to close the tunnel when you're done!"
	sleep 5 # Wait for the tunnel to be ready
	ngrok http 8000
