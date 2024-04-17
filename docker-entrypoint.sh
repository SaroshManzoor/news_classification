#!/usr/bin/env bash

if [ "$1" = "inference" ]; then
    echo "Starting streamlit server"
    poetry run python -m streamlit run news_article_classification/live_inference.py --server.port 8000 --server.address=0.0.0.0
else
    poetry run python news_article_classification/main.py
fi
