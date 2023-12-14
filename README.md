# bongGem chatbot

This project is a fine tune of GPT-2 architecture from huggingface.
In this project we used huggingface library to fine tune the model with our
custom dataset. The installation process for running the model is given below.

## Getting Started

### Prerequisites

- Python 3.x installed on your system
- If you want an fast training speed you might need cuda supported GPU.

### Usage

#### For Windows execute the commands below to activate the virtual terminal:

1. `python -m venv bongGem_venv`
2. `.\bongGem_venv\Scripts\activate`

#### For Linux execute the commands below to activate the virtual terminal:

1. `python3 -m venv bongGem_venv`
2. `source ./bongGem_venv/bin/activate`

After executing the above two commands a virtual environment will be activated.
For this project we need some requirements. We can install all the requirments
using the own command.

Now execute the following command to install all the dependencies.
`pip3 install -r requirement.txt`

By this time all the libraries will be installed, and we are ready to train
our custom model.

## Code explanation

### Dateset preprocessing

We've added two different type of dataset in our project.

* txt file based dataset
* csv file based dataset.

For txt based dataset the dataset needs to be placed in our local machine and
define the path.
However, for the csv based dataset, the dataset will be first downloaded
from a Google Drive link and automatically pre-processed for train our model.

For preprocessing the dataset we've added two classes.

* TextDataProcessor
* CSVDataProcessor

The TextDataProcessor class takes two argument
in its constructor.

* **Dataset path** - the directory of the text file
* **processed dataset path** - the directory of the processed dataset.

CSVDataProcessor also takes two argument in its
constructor.

* **Google drive id** - The file's google drive unique ID
* **processed dataset path** - the directory of the processed dataset.

Each class has one common function. `process_file_and_create_dataset` processes
dataset and create the final cleaned dataset based on the parameter.

### Model training

There is one class called `BengaliGpt` to train our model.
It takes the following argument:

* Train file path
* Model name
* Output directory
* Training batch
* Number of epoch
* Saving steps





