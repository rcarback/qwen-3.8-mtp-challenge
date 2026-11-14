# Local embeddings

The server can load a `Qwen3-Embedding-4B` checkpoint alongside the chat model.
Use a local MLX checkpoint directory that contains weights, configuration, and tokenizer files.
The server does not download files.

## Start the server

Build the server and worker with the same sources:

```sh
swift build -c release --scratch-path .build-worker --force-resolved-versions
```

Add the embedding options to your normal server command:

```sh
.build-worker/release/mlxfast-swift serve \
  --weights /path/to/chat-weights \
  --mtp-head /path/to/mtp-head \
  --embedding-weights /path/to/embedding-weights
```

The default embedding name is `qwen3-embedding-4b`.
It must differ from the chat model name.
`GET /v1/models` lists both models after startup succeeds.
Chat works as before without embedding weights. In that mode, embedding requests return status 503.

Both models stay in memory.
Allow extra memory for the embedding weights and attention buffers, which grow with input length.
The server runs one inference request at a time. Embeddings wait for any current chat request.
Embedding calls use their own model state and preserve chat prefix caches.

## Request vectors

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-4b","input":["Paris is in France.","Apples grow on trees."],"encoding_format":"float"}'
```

Each input produces a normalized vector with 2,560 values.
The response preserves input order through zero-based `index` fields.
Usage counts input tokens, including special tokens that the tokenizer adds to text.

Use `"encoding_format":"base64"` for base64 strings containing little-endian float32 bytes.
The default format is `float`.
The endpoint also accepts integer token arrays and arrays of token arrays.
Token IDs must use the embedding tokenizer's vocabulary. The server adds no tokens to these arrays.
`POST /embeddings` is an alias for the versioned route.

## Format retrieval queries

The endpoint embeds the text that you send. It does not add a chat template or retrieval instructions.
For search, include an instruction in the query text:

```text
Instruct: Retrieve the document that answers the query.
Query: What is the capital of France?
```

Send documents as plain text.
Use the same checkpoint and formatting when you build and query a vector index.
Rebuild indexes made with the previous `Qwen3-Embedding-0.6B` model before querying them with `Qwen3-Embedding-4B` vectors.
See the [model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B) for query formatting guidance.

## Limits and errors

- At most 32 inputs per request.
- At most 8,192 tokens per input and 32,768 tokens across the request.
- The worker protocol limits each serialized request to 4 MiB.
- Empty inputs, invalid token IDs, and unknown model names return status 400.
- Requests with a `dimensions` parameter return status 400.
- Inference or worker failures return status 500. A failed batch returns no partial vectors.

The server rejects inputs that exceed these limits before inference, with no truncation.
The optional `user` field has no effect.
This endpoint serves local development and does not change ranked benchmark scoring.

## Verified checkpoint

Local tests use [this MLX conversion](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ)
at revision `b5d88f1fe49b50d2ac01b4692ca2d387f14f9c72`.
The real-model test checks vector width, unit norm, batch isolation, and one retrieval ordering example.
This small test does not measure retrieval accuracy across a dataset.

Run the test with an existing local checkpoint:

```sh
MLXFAST_EMBEDDING_TEST_WEIGHTS=/path/to/embedding-weights \
  swift test --force-resolved-versions --filter EmbeddingRuntimeTests.realModelBatchIsolation
```
