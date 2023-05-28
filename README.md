<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Data Science 2023</h1> 
  <h2 align="center">Assignment 3</h2> 
  <h3 align="center">Language Analytics</h3> 


  <p align="center">
    Aleksandrs Baskakovs
  </p>
</p>


<!-- Assignment instructions -->
## Assignment instructions

Text generation is hot news right now!

For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts which do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt

<!-- ABOUT THE PROJECT -->
## About the project
This repository contains two Python scripts that enable, firstly, training of a recurrent neural newtork (RNN) on textual data, and secondly, generation of text from a user-suggested prompt using the trained model. The first script, ```train_rnn.py```, trains the model and saves it in the ```models``` folder. The second script, ```generate_text.py```, loads the saved model and generates text based on a user-suggested prompt.

<!-- Data -->
## Data
This code was originally made for the *The New York Times* data, which consist of over 2m comments made to New York Times article sections. The data can be found [here](https://www.kaggle.com/datasets/aashita/nyt-comments). The code can, however, be used with any other textual data, as long as it is saved in a ```.csv``` with a column containing the text to be used for training the model. The data should be saved in the ```data``` folder.

<!-- USAGE -->
## Usage
To use the code you need to adopt the following steps.

**NOTE:** Please note that the instructions provided here have been tested on a Mac machine running macOS Ventura 13.1, using Visual Studio Code version 1.76.0 (Universal) and a Unix-based bash terminal. While they should also be compatible with other Unix-based systems like Linux, slight variations may exist depending on the terminal and operating system you are using. To ensure a smooth installation process and avoid potential package conflicts, it is recommended to use the provided ```setup.sh``` bash file, which includes the necessary steps to create a virtual environment for the project. However, if you encounter any issues or have questions regarding compatibility on other platforms, please don't hesitate to reach out for assistance.

1. Clone repository
2. Run ``setup.sh`` in the terminal
3. Activate virtual environment
3. Run ```train_rnn.py``` in the terminal
4. Run ```generate_text.py``` in the terminal
5. Deactivate virtual environment

### Clone repository

Clone repository using the following lines in the your terminal:

```bash
git clone https://github.com/sashapustota/rnn-for-text-generation
cd rnn-for-text-generation
```

### Run ```setup.sh```

The ``setup.sh`` script is used to automate the installation of project dependencies and configuration of the environment. By running this script, you ensure consistent setup across different environments and simplify the process of getting the project up and running.

The script performs the following steps:

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the required packages
4. Deactivates the virtual environment

To run the script, run the following line in the terminal:

```bash
bash setup.sh
```

### Activate virtual environment

To activate the newly created virtual environment, run the following line in the terminal:

```bash
source ./simple-text-classification-venv/bin/activate
```

### Run ```train_rnn.py```

The ```train_rnn.py``` script perform the following steps:

1. Loads the data from the ```data``` folder
2. Preprocesses and tokenizes the data
3. Trains the rnn model
4. Saves the model and the tokenizer in the ```models``` folder

To run the script, run the following lines in the terminal:

```bash
python3 src/train_rnn.py
```

The script has the following optional arguments:

```
train_rnn.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--all_files ALL_FILES] [--common_word COMMON_WORD] [--column_name COLUMN_NAME]

options:
  -h, --help                show this help message and exit
  --epochs EPOCHS           Amount of epochs to train the model for (default: 89)
  --batch_size BATCH_SIZE   Batch size to use when training the model (default: 64)
  --all_files ALL_FILES     Select this as true if you want to load all files from your directory (default: False)
  --common_word COMMON_WORD The common word in the file names, if you are loading multiple selected files (default: "Comments")
  --column_name COLUMN_NAME The column name of the text entries in the csv file(s) (default: "commentBody")

```

An example below demonstrates how to run the script with custom arguments:

```bash
python3 src/train_rnn.py --epochs 100 --batch_size 32 --all_files True --column_name "text"
```

As the script is designed for the New York Times dataset, the default values for the optional arguments are set to the values that are most suitable for parsing of this dataset. However, if you are using a different dataset, you may want to change the default values to the ones that are most suitable for your dataset. For example, if your dataset consist of a single ```.csv``` file, you may want to set ```--all_files``` to ```True``` and ```--column_name``` to the name of the column containing the text entries in your ```.csv``` file.

### Run ```generate_text.py```

The ```generate_text.py``` script perform the following steps:

1. Loads the model and the tokenizer from the ```models``` folder
2. Generates text based on a user-suggested prompt

The script contains the following arguments:

```
generate_text.py [-h] [-p PROMPT] [-w NUM_WORDS]

options:
  -h, --help         show this help message and exit
  -p PROMPT          The prompt to start the text generation from.
  -w NUM_WORDS       The number of words to generate (default: 10)
```

As the ```PROMPT``` argument is required, the script will not run without it. An example below demonstrates how to run the script with custom arguments:

```bash
python3 src/generate_text.py -p "Never going to give you up" -w 10
```

### Deactivate virtual environment

When you are done running the scripts, deactivate the virtual environment by running the following line in the terminal:

```bash
deactivate
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:
```
│   .gitignore
│   README.md
│   requirements.txt
│   setup.sh
│       
├───data
│       .gitkeep
│       
├───utils
│       __init__.py
│       helper_functions.py
│
├───models
│       .gitkeep
│     
│
└───src
        train_rnn.py
        generate_text.py
```

<!-- REPRODUCIBILITY -->
## Reproducibility
The following results were obtained by training the model on the first 1000 comments from the New York Times dataset for 10 epochs with a batch size of 32. The model was then used to generate text based on the prompt "Never going to give you up" with number of words set to 10. The results are shown below:

```
PROMPT:  Never going to give you up
1/1 [==============================] - 0s 230ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 13ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 13ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 14ms/step
GENERATED TEXT:  Never Going To Give You Up To Be A Good Man And The Woman Is A
```