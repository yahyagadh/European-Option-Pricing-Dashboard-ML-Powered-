# Makefile for European Option Pricing project

.PHONY: help backend frontend

help:
	@echo "Available commands:"
	@echo "  make backend   - Run FastAPI backend"
	@echo "  make frontend  - Run React/Next.js frontend"
	@echo "  make all       - Run both backend and frontend"

backend:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run dev

all:
	@echo "Starting backend..."
	@$(MAKE) backend &
	@sleep 2
	@echo "Starting frontend..."
	@$(MAKE) frontend
