import re


class TextDataProcessor:
    def __init__(self, dataset_path, processed_dataset_path):
        self.dataset_path = dataset_path
        self.processed_dataset_path = processed_dataset_path

    def process_file_and_create_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            text = file.read()
            text_data = re.sub(r'\n+', '\n', text).strip()
        with open(self.processed_dataset_path, "w", encoding="utf-8") as f:
            f.write(text_data)
