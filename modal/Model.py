from transformers import TextDataset, DataCollatorForLanguageModeling


class ParentModel:
    def load_dataset(self, file_path, tokenizer, block_size=128):
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
        )
        return dataset

    def load_data_collator(self, tokenizer, mlm=False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )
        return data_collator
