IMAGE ?= docker.lazybird.app/lzai
TAG ?= latest
CONTAINER ?= lzai

FULL_IMAGE := $(IMAGE):$(TAG)

.PHONY: build push run stop restart logs shell size clean

## build: build the image
build:
	docker build -t $(FULL_IMAGE) .

## push: build and push to docker.lazybird.app
push: build
	docker push $(FULL_IMAGE)

## run: start the worker like pm2 does (CUDA_VISIBLE_DEVICES=2, autorestart,
##      30s stop timeout). First start bootstraps: S3 data sync + uv sync.
##      Override the GPU with CUDA_VISIBLE_DEVICES=1 make run
run:
	docker run -d \
		--name $(CONTAINER) \
		--gpus all \
		--restart unless-stopped \
		--stop-timeout 30 \
		-e CUDA_VISIBLE_DEVICES=$${CUDA_VISIBLE_DEVICES:-2} \
		-e LZAI_SKIP_DATA_DOWNLOAD=$${LZAI_SKIP_DATA_DOWNLOAD:-0} \
		-v lzai-data:/app/data \
		-v lzai-venv:/app/.venv \
		-v lzai-cache:/app/cache \
		$(FULL_IMAGE)

## stop: remove the container (volumes are kept)
stop:
	docker rm -f $(CONTAINER)

restart: stop run

## logs: follow container logs
logs:
	docker logs -f $(CONTAINER)

## shell: shell into the running container
shell:
	docker exec -it $(CONTAINER) bash

## size: show image size
size:
	docker image ls $(IMAGE)

## clean: remove container and named volumes (deletes synced venv + downloaded models)
clean:
	docker rm -f $(CONTAINER) || true
	docker volume rm lzai-data lzai-venv lzai-cache || true
