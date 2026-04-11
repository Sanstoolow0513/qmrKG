DOCKER_COMPOSE := docker compose
COMPOSE_FILE := docker-compose.neo4j.yml
PROJECT := qmrkg-neo4j

.PHONY: neo4j-build neo4j-up neo4j-down neo4j-down-v

neo4j-build:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -p $(PROJECT) build

neo4j-up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -p $(PROJECT) up -d

neo4j-down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -p $(PROJECT) down

neo4j-down-v:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -p $(PROJECT) down -v
