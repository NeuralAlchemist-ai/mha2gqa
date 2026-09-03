import os

import click
import torch
from rich.console import Console

from .config import GQAUserConfig
from .pipeline import GQAConversionPipeline

console = Console()
global_rank = int(os.environ.get("RANK", -1))
is_main_process = global_rank in (-1, 0)

def print_main(*args, **kwargs):
    if is_main_process:
        console.print(*args, **kwargs)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@click.group()
def cli():
    "MHA -> GQA conversion utilities"


@cli.command()
@click.option("--model-id", required=True, help="Pretrained model path or Hub id")
@click.option("--output-dir", required=True, help="Directory to write the converted GQA model")
@click.option("--kv-groups", default=8, show_default=True, help="Target number of KV groups")
@click.option("--dtype", default="float16", show_default=True, type=click.Choice(list(DTYPE_MAP)))
@click.option("--allow-non-mha", is_flag=True, default=False, help="Proceed even if source already uses GQA/MQA")
def convert(model_id, output_dir, kv_groups, dtype, allow_non_mha):
    """Convert a MHA model to GQA and save it. Does not train."""
    config = GQAUserConfig(
        model_id=model_id,
        save_path=output_dir,
        target_kv_groups=kv_groups,
        model_dtype=DTYPE_MAP[dtype],
        allow_non_mha=allow_non_mha,
    )
    print_main(f"Loading [bold]{model_id}[/bold]...")
    try:
        saved_path = GQAConversionPipeline(config).convert()
    except (NotImplementedError, ValueError) as e:
        print_main(f"[red]Conversion failed:[/red] {e}")
        raise click.Abort()

    print_main(f":white_check_mark: Saved GQA model to [green]{saved_path}[/green]")


@cli.command()
@click.option("--model-path", required=True, help="Converted GQA model path (or any causal LM)")
@click.option("--data-path", required=True, help="Local file path or HF dataset id")
@click.option("--dataset-name", default=None, help="HF dataset config name, e.g. wikitext-2-raw-v1")
@click.option("--epochs", default=1, show_default=True)
@click.option("--batch-size", default=2, show_default=True)
@click.option("--max-steps", default=None, type=int, help="Cap training steps (overrides --epochs if set)")
@click.option("--lora-output", default="./gqa_lora_output", show_default=True)
def uptrain(model_path, data_path, dataset_name, epochs, batch_size, max_steps, lora_output):
    """Run QLoRA uptraining on an existing (already-converted) model."""
    config = GQAUserConfig(
        data_path=data_path,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lora_output_dir=lora_output,
    )
    print_main(f"🔥 Starting QLoRA uptraining for [bold]{model_path}[/bold] on [bold]{data_path}[/bold]")
    try:
        ppl_pre, ppl_post = GQAConversionPipeline(config).uptrain(model_path=model_path, max_steps=max_steps)
    except Exception as e:
        print_main(f"[red]Uptraining failed:[/red] {e}")
        raise click.Abort()

    print_main(f":white_check_mark: LoRA weights saved to [green]{lora_output}[/green]")
    if ppl_pre is not None and ppl_post is not None:
        print_main(f"Perplexity: {ppl_pre:.2f} -> {ppl_post:.2f}")


@cli.command()
@click.option("--model-id", required=True)
@click.option("--output-dir", required=True, help="Where to save the converted GQA model")
@click.option("--data-path", required=True)
@click.option("--dataset-name", default=None)
@click.option("--kv-groups", default=8, show_default=True)
@click.option("--epochs", default=1, show_default=True)
@click.option("--batch-size", default=2, show_default=True)
@click.option("--max-steps", default=None, type=int)
@click.option("--lora-output", default="./gqa_lora_output", show_default=True)
@click.option("--dtype", default="float16", show_default=True, type=click.Choice(list(DTYPE_MAP)))
@click.option("--allow-non-mha", is_flag=True, default=False)
def run(model_id, output_dir, data_path, dataset_name, kv_groups, epochs,
        batch_size, max_steps, lora_output, dtype, allow_non_mha):
    """Full pipeline: convert MHA -> GQA and uptrain in one process."""
    config = GQAUserConfig(
        model_id=model_id,
        save_path=output_dir,
        data_path=data_path,
        dataset_name=dataset_name,
        target_kv_groups=kv_groups,
        epochs=epochs,
        batch_size=batch_size,
        lora_output_dir=lora_output,
        model_dtype=DTYPE_MAP[dtype],
        allow_non_mha=allow_non_mha,
    )
    print_main(f"Running full pipeline for [bold]{model_id}[/bold]...")
    try:
        ppl_pre, ppl_post = GQAConversionPipeline(config).run(max_steps=max_steps)
    except Exception as e:
        print_main(f"[red]Pipeline failed:[/red] {e}")
        raise click.Abort()

    print_main(
        f":white_check_mark: Done. GQA model → [green]{output_dir}[/green], "
        f"LoRA weights → [green]{lora_output}[/green]"
    )
    if ppl_pre is not None and ppl_post is not None:
        print_main(f"Perplexity: {ppl_pre:.2f} -> {ppl_post:.2f}")


if __name__ == "__main__":
    cli()