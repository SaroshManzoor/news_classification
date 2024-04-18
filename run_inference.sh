#!/bin/bash

# If the container name already exists, then remove & ignore errors
docker rm news_classification_inference 2> /dev/null || true


echo "Building docker image ..."
docker build -t news_classification_inference .  >/dev/null 2>&1 &&

echo " "
docker run -it --init -p 8000:8000 news_classification_inference inference
