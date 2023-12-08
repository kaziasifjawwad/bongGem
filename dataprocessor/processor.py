import re
import os


class TextDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process_file_and_create_dataset(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()
            text_data = re.sub(r'\n+', '\n', text).strip()
        with open("content/train.txt", "w", encoding="utf-8") as f:
            f.write(text_data)
