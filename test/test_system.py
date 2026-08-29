import math

from src.mha2gqa import GQAConversionPipeline, GQAUserConfig


def test_full_pipeline():
    config = GQAUserConfig(
        model_id="JackFram/llama-160m",       # real Llama arch, GQA-capable, ~160M params
        data_path="Salesforce/wikitext",
        dataset_name="wikitext-2-raw-v1",
        target_kv_groups=4,                    # 12 heads / 4 groups = 3 heads/group
        save_path="/tmp/trivial_gqa",
        lora_output_dir="./gqa_recovery_lora",
        batch_size=2,
        accumulation_steps=4,
        epochs=1
    )

    pipeline = GQAConversionPipeline(config)
    ppl_pre, ppl_post = pipeline.run(max_steps=5)

    # Primary assertion: the pipeline produced real numbers, not nan/inf.
    assert ppl_pre is not None and ppl_post is not None
    assert math.isfinite(ppl_pre) and math.isfinite(ppl_post)