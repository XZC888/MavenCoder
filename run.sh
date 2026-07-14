#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

model="deepseek-v4-flash"
dataset="humanevalplus"

key="YOUR-KEY-HERE"
url="BASE-URL-HERE"  # Optional: defaults to OpenAI official endpoint if not set

# Embedding API configuration (optional, only needed if different from main API)
embedding_key="YOUR-KEY-HERE"  # Optional: defaults to main key if not set
embedding_url="BASE-URL-HERE"  # Optional: defaults to main url if not set
embedding_model="text-embedding-3-large"

python src/main.py \
       --model $model \
       --dataset_type $dataset \
       --key $key \
       --url $url \
       --strategy prompt \
       --embedding_key $embedding_key \
       --embedding_url $embedding_url \
       --embedding_model $embedding_model \
       --verbose True
