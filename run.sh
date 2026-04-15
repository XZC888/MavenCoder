model="glm-4.7"
dataset="humanevalplus"

key="your-api-key-here"
url="your-base-url-here"  # Optional: defaults to OpenAI official endpoint if not set

# Embedding API configuration (optional, only needed if different from main API)
# For Qwen model, you should use OpenAI's embedding API
embedding_key="your-embedding-api-key-here"  # Optional: defaults to main key if not set
embedding_url="your-embedding-api-url-here"  # Optional: defaults to main url if not set
embedding_model="text-embedding-3-large"

python src/main.py \
       --model $model \
       --dataset_type $dataset \
       --key $key \
       --url $url \
       --strategy entropy
       # --embedding_key $embedding_key \
       # --embedding_url $embedding_url \
       # --embedding_model $embedding_model \
