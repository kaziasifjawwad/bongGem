from dataprocessor.processor import TextDataProcessor

if __name__ == '__main__':
    file_path = ""
    text_data_process = TextDataProcessor("test")
    text_data_process.process_file_and_create_dataset()
