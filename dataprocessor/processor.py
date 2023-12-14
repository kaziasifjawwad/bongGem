import os
import re

import gdown
import pandas as pd


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


class CSVDataProcessor:
    def __init__(self, google_drive_id, processed_dataset_path):
        self.google_drive_id = google_drive_id
        self.processed_dataset_path = processed_dataset_path

    def process_file_and_create_dataset(self):
        folder_path = 'temporaryFile'
        if folder_path not in os.listdir('.'):
            os.mkdir('temporaryFile')
        url = 'https://drive.google.com/uc?id={}'.format(self.google_drive_id)
        output = '{}/downloaded_file.csv'.format(folder_path)
        gdown.download(url, output, quiet=False)

        data = pd.read_csv(output)
        questions = data['Question'].tolist()
        answers = data['Answer'].tolist()
        with open('content/train.txt', 'w', encoding='utf-8') as file:
            for q, a in zip(questions, answers):
                file.write(f"{q}\n{a}\n")

        try:
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(folder_path)
            print(f"Folder '{folder_path}' and its contents successfully removed.")
        except OSError as e:
            print(f"Error: {folder_path} : {e.strerror}")
