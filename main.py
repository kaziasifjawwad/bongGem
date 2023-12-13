from dataprocessor.processor import TextDataProcessor, CSVDataProcessor
from models.gpt import BengaliGpt

if __name__ == '__main__':
    custom_data_process = CSVDataProcessor(
        google_drive_id="15mfX5bI3aGdAPsVoegt6aHcbuYAspiIf",
        processed_dataset_path="content/train.txt"
    )
    custom_data_process.process_file_and_create_dataset()
    train_file_path = custom_data_process.processed_dataset_path
    model_name = 'flax-community/gpt2-bengali'
    output_dir = 'custom_q_and_a'
    overwrite_output_dir = False
    per_device_train_batch_size = 8
    num_train_epochs = 50.0
    save_steps = 50000

    bengali_gpt = BengaliGpt(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

    bengali_gpt.train()
