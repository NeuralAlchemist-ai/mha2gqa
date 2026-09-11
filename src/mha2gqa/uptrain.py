import os
import logging
import math

import torch
from tqdm import tqdm

from .estimator import estimate_uptraining_time, estimate_vram

logger = logging.getLogger(__name__)

class GQAUptrain:
    def __init__(self, user_config, dataset, eval_dataset, tokenizer, model_or_path):
        self.rank = user_config.lora_rank
        self.alpha = user_config.lora_alpha
        self.dropout = user_config.lora_dropout
        self.target_modules = user_config.lora_target
        self.lora_output_dir = user_config.lora_output_dir

        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.batch_size = user_config.batch_size
        self.accumulation_steps = user_config.accumulation_steps
        self.lr = user_config.lr
        self.epochs = user_config.epochs

        self.model_dtype = getattr(user_config, "model_dtype", torch.float16)

        # Detect DDP environment variables (set by torchrun / accelerate / Hugging Face Trainer)
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.global_rank = int(os.environ.get("RANK", -1))
        self.is_main_process = self.global_rank in (-1, 0)

        # If a string is passed — it's a path/repo id to load via setup_qlora().
        # If an object is passed — it's an already-instantiated model.
        if isinstance(model_or_path, str):
            self.model_path = model_or_path
            self.model = None
        else:
            self.model_path = None
            self.model = model_or_path

    def setup_qlora(self):
        # Lazy imports for optional heavy dependencies.
        from transformers import AutoModelForCausalLM

        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception:
            LoraConfig = get_peft_model = prepare_model_for_kbit_training = None

        # 4-bit quantization requires both bitsandbytes/transformers support *and*
        # an actual CUDA device — bitsandbytes' kernels don't run on CPU.
        self.can_quantize = BitsAndBytesConfig is not None and torch.cuda.is_available()

        if self.model_path is not None and self.model is None:
            if self.local_rank != -1:
                device_target = f"cuda:{self.local_rank}"
                device_map = {"": self.local_rank}
            elif torch.cuda.is_available():
                device_target = "cuda:0"
                device_map = {"": 0}
            else:
                device_target = "cpu"
                device_map = None

            if self.can_quantize:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.model_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                if self.is_main_process:
                    logger.info(f"Loading and 4-bit quantizing GQA model from {self.model_path}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, quantization_config=bnb_config, device_map=device_map
                )
            else:
                if self.is_main_process:
                    logger.info(f"Loading model without quantization from {self.model_path}...")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                if torch.cuda.is_available():
                    self.model = self.model.to(device_target)

        # Auto-detect target modules for LoRA if not explicitly configured.
        if not self.target_modules:
            try:
                param_keys = [n for n, _ in self.model.named_parameters()]
            except Exception:
                param_keys = list(self.model.state_dict().keys())

            detected = set()
            for key in param_keys:
                if "q_proj" in key:
                    detected.add("q_proj")
                if "k_proj" in key:
                    detected.add("k_proj")
                if "v_proj" in key:
                    detected.add("v_proj")
                if "o_proj" in key:
                    detected.add("o_proj")
                if "gate_proj" in key:
                    detected.add("gate_proj")
                if "up_proj" in key:
                    detected.add("up_proj")
                if "down_proj" in key:
                    detected.add("down_proj")
                if "c_attn" in key:
                    detected.add("c_attn")
                if "c_proj" in key:
                    detected.add("c_proj")
                if "c_fc" in key:
                    detected.add("c_fc")

            if {"q_proj", "k_proj", "v_proj"}.intersection(detected):
                self.target_modules = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if m in detected]
            elif {"c_attn", "c_proj"}.intersection(detected):
                self.target_modules = [m for m in ("c_attn", "c_proj", "c_fc") if m in detected]
            else:
                self.target_modules = list(self.target_modules or [])

        if LoraConfig is None:
            raise ImportError("peft is not installed — install it to use LoRA/QLoRA uptraining.")

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.can_quantize:
            self.model = prepare_model_for_kbit_training(self.model)

        self.peft_model = get_peft_model(self.model, lora_config)
        if self.is_main_process:
            self.peft_model.print_trainable_parameters()

        return self.peft_model

    @staticmethod
    def _calculate_perplexity(model, eval_dataset, max_length=256, is_main_process=True):
        """
        Per-example perplexity over a tokenized HF `Dataset`.

        Deliberately NOT a sliding-window/concatenated-document implementation:
        eval_dataset here is a `Dataset` of separate tokenized examples
        (eval_dataset["input_ids"] -> list[list[int]]), not one long tensor,
        so each example is scored independently and averaged.
        """
        model.eval()
        total_nll = 0.0
        total_tokens = 0

        if "input_ids" not in eval_dataset.column_names:
            raise ValueError("eval_dataset is missing an 'input_ids' column")

        input_ids_list = eval_dataset["input_ids"]
        if is_main_process:
            logger.info(f"Evaluating perplexity on {len(input_ids_list)} examples...")

        disable_tqdm = not is_main_process

        with torch.no_grad():
            for i in tqdm(range(len(input_ids_list)), desc="Evaluating perplexity", disable=disable_tqdm):
                input_ids = torch.tensor([input_ids_list[i]], device=model.device)
                input_ids = input_ids[:, :max_length]

                # Next-token loss needs at least 2 tokens (shifted logits/labels
                # would otherwise be length 0, which silently produces nan).
                if input_ids.size(1) < 2:
                    continue

                target_ids = input_ids.clone()

                try:
                    outputs = model(input_ids, labels=target_ids)
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    # Skip degenerate samples instead of letting one nan poison
                    # the running total for every subsequent example.
                    continue

                total_nll += (loss * input_ids.size(1)).item()
                total_tokens += input_ids.size(1)

        if total_tokens == 0:
            return float("inf")

        avg_nll = total_nll / total_tokens
        return math.exp(avg_nll)

    def train(self, max_steps=None):
        is_bf16 = self.model_dtype == torch.bfloat16
        is_fp16 = self.model_dtype == torch.float16
        seq_len_for_estimation = 256

        from transformers import (
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        # paged_adamw_8bit needs bitsandbytes' CUDA kernels; fall back to a
        # plain optimizer on CPU (e.g. CI) or when quantization isn't active.
        optim = "paged_adamw_8bit" if getattr(self, "can_quantize", False) else "adamw_torch"

        training_args = TrainingArguments(
            output_dir=self.lora_output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accumulation_steps,
            learning_rate=self.lr,
            num_train_epochs=self.epochs,
            max_steps=max_steps if max_steps is not None else -1,
            logging_steps=10,
            save_strategy="epoch",
            gradient_checkpointing=True,
            bf16=is_bf16,
            fp16=is_fp16,
            optim=optim,
            report_to="none",
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=self.peft_model,
            train_dataset=self.dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        ppl_pre = None
        ppl_post = None

        if self.eval_dataset is not None:
            if self.is_main_process:
                logger.info("Pre-Uptraining Perplexity Evaluation...")
            try:
                ppl_pre = self._calculate_perplexity(
                    self.peft_model, self.eval_dataset, seq_len_for_estimation, is_main_process=self.is_main_process
                )
                if self.is_main_process:
                    logger.info(f"Pre-Uptraining Perplexity: {ppl_pre:.2f}")
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f" Could not calculate pre-training perplexity: {e}")
        else:
            if self.is_main_process:
                print(
                    "[info] No eval_dataset provided — skipping perplexity check. "
                    "Pass a raw-text held-out dataset (e.g. wikitext-2 test split) to measure quality."
                )

        if self.is_main_process:
            logger.info("Estimating VRAM usage and uptraining time...")
            num_train_samples = len(self.dataset)
            world_size = getattr(training_args, "world_size", 1) or 1
            effective_batch_size = (
                training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
            )
            steps_per_epoch = num_train_samples / effective_batch_size
            num_steps = max_steps if max_steps is not None else int(steps_per_epoch * training_args.num_train_epochs)

            print(
                estimate_vram(
                    self.model, self.peft_model, training_args.per_device_train_batch_size,
                    seq_len_for_estimation, compute_dtype=self.model_dtype,
                ).summary()
            )
            logger.info(estimate_uptraining_time(trainer, num_steps).summary())

            logger.info("Starting uptraining (accuracy recovery)...")

        trainer.train()

        if self.eval_dataset is not None:
            try:
                ppl_post = self._calculate_perplexity(
                    self.peft_model, self.eval_dataset, seq_len_for_estimation, is_main_process=self.is_main_process
                )
                if self.is_main_process:
                    logger.info(f"Post-Uptraining Perplexity: {ppl_post:.2f}")
                    if ppl_pre is not None:
                        print(
                            f"Recovery: {ppl_pre:.2f} -> {ppl_post:.2f} "
                            f"({'improved' if ppl_post < ppl_pre else 'did not improve'})"
                        )
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f" Could not calculate post-training perplexity: {e}")

        if self.is_main_process:
            logger.info(f"Saving LoRA weights to {self.lora_output_dir}")
        trainer.save_model(self.lora_output_dir)
        return ppl_pre, ppl_post