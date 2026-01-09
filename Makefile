.PHONY: build run stop clean test deploy

build:
	docker build -t doc-intelligence .

run:
	docker run -d \
		-p 8000:8000 \
		--env-file .env \
		-v chroma_data:/app/chroma_db \
		--name doc-intelligence \
		doc-intelligence

dev:
	docker-compose up --build

stop:
	docker stop doc-intelligence || true
	docker rm doc-intelligence || true

clean:
	docker system prune -f

test:
	curl http://localhost:8000/health

logs:
	docker logs -f doc-intelligence

deploy:
	@echo "Deploying to production..."
	