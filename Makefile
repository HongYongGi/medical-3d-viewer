.PHONY: run run-renderer run-all install dev clean

run:
	streamlit run src/medical_viewer/app.py --server.port=8501

run-renderer:
	cd go_renderer && go run cmd/renderer/main.go

run-all:
	@echo "Start in two terminals:"
	@echo "  Terminal 1: make run"
	@echo "  Terminal 2: make run-renderer"

install:
	pip install -e ".[nnunet]"

dev:
	pip install -e ".[dev]"

build-renderer:
	go build -o bin/renderer go_renderer/cmd/renderer/main.go

clean:
	rm -rf data/uploads/* data/results/* data/meshes/*

go-init:
	go mod tidy
