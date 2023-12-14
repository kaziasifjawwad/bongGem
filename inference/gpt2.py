from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPTInference:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def load_model(self, model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model

    def load_tokenizer(self, tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def generate_text(self, sequence, max_length):
        model = self.load_model(self.model_file_path)
        tokenizer = self.load_tokenizer(self.model_file_path)
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
