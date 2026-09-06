import random
import numpy as np
import torch
from transformers import AutoTokenizer, set_seed as hf_set_seed

from .config import GQAUserConfig
from .converter import GQAConverter
from .data import DatasetLoader
from .detector import AutoArchitectureDetector
from .grouping.calibration import Calibration
from .model_loader import ModelLoader
from .uptrain import GQAUptrain


def set_global_seed(seed: int = 42):
    """Sets seed across python random, numpy, torch CPU/CUDA, and transformers for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


class GQAConversionPipeline:
    def __init__(self, config: GQAUserConfig):
        self.config = config
        seed = getattr(config, "seed", 42)
        set_global_seed(seed)

    def _load_model(self):
        model, hf_config, tokenizer = ModelLoader(
            self.config.model_id,
            allow_non_mha=getattr(self.config, "allow_non_mha", False),
        ).load()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, hf_config, tokenizer

    def _load_and_tokenize_data(self, tokenizer):
        return DatasetLoader(tokenizer, max_length=256).load(
            self.config.data_path, name=self.config.dataset_name
        )

    def _detect_architecture(self, model, hf_config):
        detector = AutoArchitectureDetector(hf_config, model)
        return detector.dynamic_search()

    def _calibrate(self, calibration_data, model, hf_config):
        calibrator = Calibration(hf_config, self.config)
        return calibrator.calibrate(model, calibration_data)

    def _get_grouping_if_available(self, model, hf_config, tokenizer):
        """Runs calibration if data_path is provided, returning cluster grouping or None."""
        if not self.config.data_path:
            return None

        train_data, _ = self._load_and_tokenize_data(tokenizer)
        num_samples = min(32, len(train_data))
        calib_samples = [train_data[i] for i in range(num_samples)]
        return self._calibrate(calib_samples, model, hf_config)

    def _convert(self, model, hf_config, mapping, grouping=None):
        converter = GQAConverter(model, hf_config, self.config, grouping=grouping)
        converter.reconfig(mapping)
        converter.save_gqa_model(hf_config)

    def _uptrain(self, train_data, eval_data, tokenizer, model_or_path, max_steps=None):
        uptrainer = GQAUptrain(self.config, train_data, eval_data, tokenizer, model_or_path=model_or_path)
        uptrainer.setup_qlora()
        return uptrainer.train(max_steps=max_steps)

    def convert(self):
        """Load a source model, convert MHA -> GQA, save it. No training."""
        import os
        global_rank = int(os.environ.get("RANK", -1))
        model, hf_config, tokenizer = self._load_model()
        mapping = self._detect_architecture(model, hf_config)
        grouping = self._get_grouping_if_available(model, hf_config, tokenizer)
        self._convert(model, hf_config, mapping, grouping=grouping)
        # save tokenizer alongside the converted model so `uptrain` can reload it standalone (rank 0 only)
        if global_rank in (-1, 0):
            tokenizer.save_pretrained(self.config.save_path)
        return self.config.save_path

    def uptrain(self, model_path: str = None, max_steps: int = None):
        """Run QLoRA uptraining against an already-converted (or any) model path."""
        path = model_path or self.config.save_path
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_data, eval_data = self._load_and_tokenize_data(tokenizer)
        return self._uptrain(train_data, eval_data, tokenizer, model_or_path=path, max_steps=max_steps)

    def run(self, max_steps: int = None):
        """Full pipeline: convert, then immediately uptrain in the same process."""
        model, hf_config, tokenizer = self._load_model()
        train_data, eval_data = self._load_and_tokenize_data(tokenizer)
        mapping = self._detect_architecture(model, hf_config)
        
        num_samples = min(32, len(train_data))
        calib_samples = [train_data[i] for i in range(num_samples)]
        grouping = self._calibrate(calib_samples, model, hf_config)

        self._convert(model, hf_config, mapping, grouping=grouping)
        return self._uptrain(train_data, eval_data, tokenizer, model_or_path=self.config.save_path, max_steps=max_steps)