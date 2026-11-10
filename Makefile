.PHONY: install test lint benchmark benchmark-fast train serve docker-build docker-run

install:        ## Install with development extras
	pip install -e ".[dev,api]"

test:           ## Run the test suite
	pytest -v

lint:           ## Lint with ruff
	ruff check .

benchmark:      ## Full benchmark: 3 seeds, random vs scaffold split (~30-60 min)
	admet benchmark --seeds 3

benchmark-fast: ## Quick benchmark with smaller models (~5 min)
	admet benchmark --seeds 1 --fast

train:          ## Select, evaluate and save the production models
	admet train

serve:          ## Run the API on http://localhost:8080 (docs at /docs)
	uvicorn api.app:app --port 8080 --reload

docker-build:   ## Build the API image (trains models during the build)
	docker build -t admet-predictor .

docker-run:     ## Run the API image
	docker run --rm -p 8080:8080 admet-predictor
