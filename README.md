# mha2gqa

Convert Multi-Head Attention (MHA) language models to Grouped-Query Attention (GQA), then recover accuracy with QLoRA uptraining — as a CLI, a Python API, or both.

GQA reduces the number of key/value heads a model uses, shrinking the KV cache and speeding up inference, at the cost of some quality. `gqa-forge` automates the conversion (mean-pooling K/V heads into groups) and gives you a fast QLoRA recovery loop to measure and recoup that quality loss, with VRAM and runtime estimates before you commit to a full run.

## Features

- **Architecture-aware conversion** — detects K/V projection layout from a model's config and state dict (registry-based for known families, heuristic fallback for others).
- **Safe by default** — refuses to compress a model that's already GQA/MQA unless explicitly overridden.
- **QLoRA uptraining** — 4-bit quantized fine-tuning on GPU to recover perplexity lost during conversion, with automatic fallback to unquantized LoRA on CPU.
- **Resource estimation** — VRAM and training-time estimates before you launch a full run.
- **CLI and Python API** — use it as a standalone tool or embed it in your own pipeline.

## Supported architectures

Registry-backed support for models with separate `q_proj`/`k_proj`/`v_proj` projections:

- Llama (Llama 1/2, CodeLlama, Vicuna, Alpaca)
- Mistral, Mixtral
- Qwen2
- Gemma, Gemma 2
- StableLM
- GPT-J

Unrecognized architectures with separate Q/K/V layers fall back to heuristic detection. Fused-QKV architectures (GPT-2, GPT-NeoX/Pythia, Falcon, Phi-3) are **not currently supported** — conversion requires separable K/V weight matrices, and the tool will raise a clear error rather than silently doing the wrong thing.

## Installation

```bash
# Conversion only (lightweight — no CUDA-specific dependencies)
pip install gqa-forge

# With QLoRA uptraining support
pip install gqa-forge[train]

# Development setup
git clone https://github.com/<your-org>/gqa-forge.git
cd gqa-forge
pip install -e ".[train,dev]"
```

Requires Python 3.10+. QLoRA uptraining requires a CUDA GPU for 4-bit quantization; without one, uptraining falls back to standard (unquantized) LoRA.

## CLI usage

### Convert a model

```bash
mha2gqa convert \
  --model-id openlm-research/open_llama_3b \
  --output-dir ./gqa_model_output \
  --kv-groups 8 \
  --dtype float16
```

### Uptrain a converted model

```bash
mha2gqa uptrain \
  --model-path ./gqa_model_output \
  --data-path Salesforce/wikitext \
  --dataset-name wikitext-2-raw-v1 \
  --epochs 1 \
  --batch-size 4 \
  --lora-output ./gqa_recovery_lora
```

### Full pipeline (convert + uptrain in one run)

```bash
mha2gqa run \
  --model-id openlm-research/open_llama_3b \
  --output-dir ./gqa_model_output \
  --data-path Salesforce/wikitext \
  --dataset-name wikitext-2-raw-v1 \
  --kv-groups 8 \
  --epochs 1 \
  --lora-output ./gqa_recovery_lora
```

Run `mha2gqa <command> --help` for the full list of options on any command.

## Python API

```python
from mha2gqa import GQAUserConfig, GQAConversionPipeline

config = GQAUserConfig(
    model_id="openlm-research/open_llama_3b",
    data_path="Salesforce/wikitext",
    dataset_name="wikitext-2-raw-v1",
    target_kv_groups=8,
    save_path="/tmp/trivial_gqa",
    lora_output_dir="./gqa_recovery_lora",
    batch_size=4,
    accumulation_steps=4,
    epochs=1,
)

pipeline = GQAConversionPipeline(config)
ppl_pre, ppl_post = pipeline.run()

print(f"Perplexity: {ppl_pre:.2f} -> {ppl_post:.2f}")
```

Or run conversion and uptraining as separate steps:

```python
pipeline = GQAConversionPipeline(config)
gqa_model_path = pipeline.convert()          # convert and save, no training
ppl_pre, ppl_post = pipeline.uptrain(model_path=gqa_model_path)
```

See [`experiments/complete_example.ipynb`](experiments/complete_example.ipynb) for a runnable end-to-end walkthrough.

## How it works

1. **Load** — fetches the source model and validates it's genuinely MHA (`num_key_value_heads == num_attention_heads`) before downloading full weights.
2. **Detect** — maps each transformer layer's K/V projection tensors, using a known-architecture registry or a heuristic fallback.
3. **Convert** — mean-pools each group of attention heads' K/V weights down to `target_kv_groups`, producing a smaller KV cache footprint.
4. **Save** — reconstructs the model with the reduced KV head count and saves it as a standard Hugging Face checkpoint.
5. **Uptrain** — optionally fine-tunes the converted model with QLoRA on a held-out dataset, reporting perplexity before and after to quantify recovery.

## Project structure

```
src/mha2gqa/
├── architectures/     # per-family K/V layer mappings and adapter interface
├── cli.py             # command-line interface (convert / uptrain / run)
├── config.py          # GQAUserConfig — pipeline configuration
├── converter.py        # MHA -> GQA weight conversion
├── data.py             # dataset loading and tokenization
├── detector.py          # architecture detection
├── estimator.py         # VRAM and training-time estimation
├── model_loader.py      # model loading and MHA validation
├── pipeline.py          # end-to-end orchestration
└── uptrain.py           # QLoRA uptraining and perplexity evaluation

test/                    # pytest suite
experiments/             # example notebooks
```

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

The test suite includes unit tests for weight-conversion math and architecture detection, CLI command tests, and an end-to-end pipeline smoke test using a small public model.

## Limitations

- Fused-QKV architectures are not supported (see above).
- QLoRA uptraining's 4-bit quantization path requires a CUDA GPU; CPU runs fall back to unquantized LoRA, which is slower and more memory-intensive.
- K/V head compression is mean-pooling based; other merging strategies (e.g. selecting a representative head, learned merging) are not currently implemented.

## License

MIT — see [LICENSE](LICENSE).