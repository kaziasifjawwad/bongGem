from dataprocessor.processor import TextDataProcessor
from modal.gpt import BengaliGpt

if __name__ == '__main__':
    text_data_process = TextDataProcessor(dataset_path="test",
                                          processed_dataset_path="content/train.txt")
    text_data_process.process_file_and_create_dataset()

    train_file_path = text_data_process.dataset_path
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
