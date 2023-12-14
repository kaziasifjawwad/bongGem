from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from models.model import ParentModel


class BengaliGpt(ParentModel):
    def __init__(self,
                 train_file_path,
                 model_name,
                 output_dir,
                 overwrite_output_dir,
                 per_device_train_batch_size,
                 num_train_epochs,
                 save_steps
                 ):
        self.train_file_path = train_file_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.save_steps = save_steps

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_dataset = super().load_dataset(self.train_file_path, tokenizer)
        data_collator = super().load_data_collator(tokenizer)
        tokenizer.save_pretrained(self.output_dir)

        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        model.save_pretrained(self.output_dir)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=self.overwrite_output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        trainer.train()
        trainer.save_model()
