import logging
import os

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

class DatasetLoader:

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load(self, data_path_or_name: str, name) -> Dataset:
        """
        Universal loader.
        Accepts: local file path (.json, .jsonl, .txt)
        or Hugging Face repository name (e.g., 'openwebtext' or 'imdb').
        """
        # Scenario 1: Local file
        if os.path.exists(data_path_or_name):
            ext = os.path.splitext(data_path_or_name)[1].lower()
            
            if ext == '.jsonl' or ext == '.json':
                dataset = load_dataset('json', data_files=data_path_or_name, split='train')
                eval_dataset = load_dataset('json', data_files=data_path_or_name, split='train[:10%]')
            elif ext == '.txt':
                dataset = load_dataset('text', data_files=data_path_or_name, split='train')
                eval_dataset = load_dataset('text', data_files=data_path_or_name, split='train[:10%]')
            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .json, .jsonl, or .txt")
            
            logger.info("Successfully loaded local dataset: %s (%d rows)", data_path_or_name, len(dataset))
        
        # Scenario 2: Load from Hugging Face Hub
        else:
            logger.info("Local file not found. Attempting to download from Hugging Face Hub: %s...", data_path_or_name)
            dataset = load_dataset(data_path_or_name, name, split='train')
            eval_dataset = load_dataset(data_path_or_name, name, split='train[:10%]')
            
        return self._prepare_and_tokenize(dataset, eval_dataset)

    def _prepare_and_tokenize(self, dataset: Dataset, eval_dataset: Dataset) -> Dataset:
        """
        Normalizes the data structure and tokenizes it.
        """
        column_names = dataset.column_names
        
        def tokenize_function(examples):
            if "text" in column_names:
                texts = examples["text"]
            elif "instruction" in column_names and "output" in column_names:
                texts = [
                    f"Instruction: {ins}\nResponse: {out}" 
                    for ins, out in zip(examples["instruction"], examples["output"])
                ]
            else:
                first_col = column_names[0]
                texts = examples[first_col]

            outputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            return outputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Tokenizing dataset"
        )
        eval_tokenized_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Tokenizing eval dataset"
        )
        
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)
        tokenized_eval_dataset = eval_tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)

        return tokenized_dataset, tokenized_eval_dataset
