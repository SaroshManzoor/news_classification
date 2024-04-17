#!/bin/bash

SECONDS=0

# If the container name already exists, then remove & ignore errors
docker rm news_article_classification_pipeline 2> /dev/null || true


echo "Building docker image ..."
docker build -t news_article_classification_pipeline .  >/dev/null 2>&1 &&

echo " "
printf "\nRunning pipeline. This will take a few minutes.\n"
docker run --name news_article_classification_pipeline -it news_article_classification_pipeline &&

printf "\n Dumping predictions and trained model to storage directory \n"
container_id=$(docker ps -aqf 'name=news_article_classification_pipeline')
docker cp "$container_id":/src/storage/data/predictions.json storage/data/predictions.json
docker cp "$container_id":/src/storage/model_registry/. storage/model_registry


duration=$SECONDS
echo""
echo "Execution took $((duration / 60)) minutes and $((duration % 60)) seconds."

