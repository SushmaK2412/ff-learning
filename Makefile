.PHONY: build up down logs clean fresh-data help

# Default target
help:
	@echo "Federated Learning Financial Forecasting - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all containers in detached mode"
	@echo "  make logs       - Follow orchestrator logs"
	@echo "  make logs-all   - Follow all container logs"
	@echo "  make down       - Stop and remove all containers"
	@echo "  make clean      - Remove images, volumes, and caches"
	@echo "  make fresh-data - Clean old data and download fresh data"
	@echo "  make fresh      - Clean everything and start fresh (clean + fresh-data + build + up)"

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build

# Start containers
up:
	@echo "🚀 Starting containers..."
	docker-compose up -d
	@echo "✅ Containers started. Dashboard available at http://localhost:8501"

# View logs
logs:
	@echo "📋 Following orchestrator logs (Ctrl+C to exit)..."
	docker-compose logs -f orchestrator

# View all logs
logs-all:
	@echo "📋 Following all container logs (Ctrl+C to exit)..."
	docker-compose logs -f

# Stop containers
down:
	@echo "🛑 Stopping containers..."
	docker-compose down
	@echo "✅ Containers stopped"

# Clean everything
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup complete"

# Clean old data files and download fresh
fresh-data:
	@echo "🗑️  Cleaning old data files..."
	@rm -rf data/raw/*.csv data/processed/*.csv data/metadata.json 2>/dev/null || true
	@echo "✅ Old data files removed"
	@echo "📥 Downloading fresh financial data..."
	@python3 data/download_data.py
	@echo "✅ Fresh data downloaded"

# Complete fresh start
fresh: clean fresh-data build up
	@echo ""
	@echo "✨ Fresh start complete!"
	@echo "🌐 Dashboard: http://localhost:8501"

