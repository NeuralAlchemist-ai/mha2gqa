import time
from dataclasses import dataclass

import torch

BYTES_PER_DTYPE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
}

# NF4 with double quantization (bitsandbytes default for QLoRA) averages
# ~4.127 bits/param rather than a clean 4 bits, due to the quantization
# constants themselves needing storage. Source: QLoRA paper, Dettmers et al. 2023.
NF4_DOUBLE_QUANT_BITS_PER_PARAM = 4.127


@dataclass
class VRAMEstimate:
    base_model_gb: float
    lora_params_gb: float
    optimizer_state_gb: float
    gradients_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    trainable_params: int
    total_params: int
    fits_in_available: bool | None = None
    available_gb: float | None = None

    def summary(self) -> str:
        lines = [
            f"Base model (4-bit NF4, {self.total_params/1e9:.2f}B params): {self.base_model_gb:.2f} GB",
            f"LoRA adapter weights ({self.trainable_params/1e6:.1f}M trainable):     {self.lora_params_gb:.3f} GB",
            f"Optimizer state (8-bit AdamW):                    {self.optimizer_state_gb:.3f} GB",
            f"Gradients (LoRA params only):                     {self.gradients_gb:.3f} GB",
            f"Activations (approx., checkpointing assumed on):  {self.activations_gb:.2f} GB",
            f"Overhead (CUDA context, fragmentation, ~15%):     {self.overhead_gb:.2f} GB",
            "-" * 60,
            f"ESTIMATED TOTAL:                                  {self.total_gb:.2f} GB",
        ]
        if self.available_gb is not None:
            verdict = "✓ should fit" if self.fits_in_available else "✗ likely to OOM"
            lines.append(f"Available VRAM: {self.available_gb:.2f} GB  ->  {verdict}")
        return "\n".join(lines)


def estimate_vram(
    model,
    peft_model,
    batch_size: int,
    seq_len: int,
    compute_dtype: torch.dtype = torch.float16,
    optimizer_bytes_per_param: float = 2.0,  # paged_adamw_8bit: ~2 bytes/param
    gradient_checkpointing: bool = True,
    device_index: int = 0,
) -> VRAMEstimate:
    """
    Estimate peak VRAM for a QLoRA uptraining run.

    `model` should be the base (already 4-bit quantized) model.
    `peft_model` should be the model *after* get_peft_model() has wrapped it,
    so trainable-parameter counts are exact rather than guessed.
    """
    total_params = sum(p.numel() for p in model.parameters())
    base_model_gb = (total_params * NF4_DOUBLE_QUANT_BITS_PER_PARAM / 8) / 1e9

    trainable_params, _ = peft_model.get_nb_trainable_parameters()
    dtype_bytes = BYTES_PER_DTYPE.get(compute_dtype, 2)

    lora_params_gb = (trainable_params * dtype_bytes) / 1e9
    optimizer_state_gb = (trainable_params * optimizer_bytes_per_param) / 1e9
    gradients_gb = (trainable_params * dtype_bytes) / 1e9

    # Activation memory: Megatron-style formula (Korthikanti et al. 2022),
    # sbh(34 + 5*a*s/h) bytes per layer without recomputation. With gradient
    # checkpointing (the default from prepare_model_for_kbit_training), only
    # layer inputs are retained between recompute boundaries, which cuts this
    # to roughly a small constant multiple of s*b*h per layer instead.
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", 4096)
    num_layers = getattr(config, "num_hidden_layers", 32)
    num_heads = getattr(config, "num_attention_heads", 32)

    if gradient_checkpointing:
        # Conservative constant for checkpointed activations (inputs retained
        # per layer boundary only). This is a rough multiplier, not exact —
        # validate against the empirical dry-run below before trusting it.
        per_token_per_layer_bytes = 4 * hidden_size
    else:
        per_token_per_layer_bytes = hidden_size * (34 + 5 * num_heads * seq_len / hidden_size)

    activations_gb = (num_layers * seq_len * batch_size * per_token_per_layer_bytes) / 1e9

    subtotal = base_model_gb + lora_params_gb + optimizer_state_gb + gradients_gb + activations_gb
    overhead_gb = subtotal * 0.15  # CUDA context + allocator fragmentation
    total_gb = subtotal + overhead_gb

    available_gb = None
    fits = None
    if torch.cuda.is_available():
        available_gb = torch.cuda.get_device_properties(device_index).total_memory / 1e9
        fits = total_gb < available_gb * 0.9  # leave 10% margin

    return VRAMEstimate(
        base_model_gb=base_model_gb,
        lora_params_gb=lora_params_gb,
        optimizer_state_gb=optimizer_state_gb,
        gradients_gb=gradients_gb,
        activations_gb=activations_gb,
        overhead_gb=overhead_gb,
        total_gb=total_gb,
        trainable_params=trainable_params,
        total_params=total_params,
        fits_in_available=fits,
        available_gb=available_gb,
    )


@dataclass
class TimeEstimate:
    num_steps: int
    seconds_per_step: float
    total_seconds: float
    method: str  # "measured" or "analytical (unverified)"

    def summary(self) -> str:
        h, rem = divmod(self.total_seconds, 3600)
        m, s = divmod(rem, 60)
        return (
            f"Steps: {self.num_steps}  |  {self.seconds_per_step:.2f} s/step "
            f"({self.method})\n"
            f"Estimated duration: {int(h)}h {int(m)}m {int(s)}s"
        )


def calibrate_seconds_per_step(trainer, num_warmup_steps: int = 3) -> float:
    """
    The trustworthy way to estimate step time: actually run a few real steps
    and measure them, rather than guessing from a spec sheet. Sequence length,
    LoRA rank, flash-attention availability, and CUDA kernel warm-up all move
    this number more than a hardware-name lookup table ever will.
    """
    train_dataloader = trainer.get_train_dataloader()
    model = trainer.model
    model.train()

    step_times = []
    data_iter = iter(train_dataloader)
    for _ in range(num_warmup_steps):
        batch = next(data_iter)
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        outputs = model(**batch)
        outputs.loss.backward()
        model.zero_grad()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_times.append(time.perf_counter() - start)

    # Drop the first measurement — first-step CUDA kernel compilation/warm-up
    # is not representative of steady-state throughput.
    steady_state = step_times[1:] if len(step_times) > 1 else step_times
    return sum(steady_state) / len(steady_state)


def estimate_uptraining_time(
    trainer,
    num_steps: int,
    seconds_per_step: float | None = None,
    calibrate: bool = True,
) -> TimeEstimate:
    """
    If seconds_per_step isn't provided and calibrate=True, runs a short
    real calibration (see calibrate_seconds_per_step). Otherwise falls back
    to a rough analytical guess — clearly labeled as unverified, since it's
    a placeholder, not a measurement.
    """
    if seconds_per_step is not None:
        method = "provided"
    elif calibrate:
        seconds_per_step = calibrate_seconds_per_step(trainer)
        method = "measured (3-step calibration)"
    else:
        seconds_per_step = 1.0  # crude placeholder — replace with calibrate=True when possible
        method = "analytical (unverified — treat as a placeholder, not a real estimate)"

    return TimeEstimate(
        num_steps=num_steps,
        seconds_per_step=seconds_per_step,
        total_seconds=num_steps * seconds_per_step,
        method=method,
    )