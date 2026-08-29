import logging

logger = logging.getLogger(__name__)

def is_mha_architecture(config) -> bool:
    """
    Returns True if a model's attention is standard Multi-Head Attention
    (i.e. NOT already GQA/MQA). Works from config alone — no weights needed —
    so this can run before downloading a multi-GB checkpoint.

    Absence of num_key_value_heads means the architecture (or the installed
    transformers version) predates GQA entirely, which is equivalent to MHA
    for our purposes.
    """
    num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError(
            "Config has no 'num_attention_heads' — can't determine whether "
            "this is an attention-based causal LM at all."
        )

    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        return True

    return num_kv_heads == num_heads

class ModelLoader:

    def __init__(self, model_path: str, allow_non_mha: bool = False):
        self.model_path = model_path
        self.allow_non_mha = allow_non_mha

    def load(self):
        """
        Loads the model from a local path or the Hugging Face Hub.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        config = AutoConfig.from_pretrained(self.model_path)

        # Cheap check first — catches a non-MHA source before downloading weights.
        if is_mha_architecture(config):
            logger.info("Architecture %s supports MHA. Continuing with load...", config.model_type)

            logger.info("Loading model from %s...", self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else: 
            if not self.allow_non_mha:
                raise ValueError(
                    f"Architecture {config.model_type} already uses GQA/MQA (num_key_value_heads != num_attention_heads). "
                    "To compress further anyway, set allow_non_mha=True."
                )
            logger.info("Architecture %s already uses GQA/MQA. allow_non_mha=True, continuing...", config.model_type)
            logger.info("Loading model from %s...", self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        logger.info("Model loaded successfully: %s", self.model_path)
        return model, config, tokenizer
