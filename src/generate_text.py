# making sure that tensorflow doesn't clutter the terminal with info
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# import argparse for command line arguments
import argparse

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(420)

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# import helper functions
import sys
sys.path.append(os.getcwd())
import utils.helper_functions as hf

# load tokenizer
from joblib import load

# import the max sequence length we determined in the training script from a .txt file
def get_max_sequence_len():
    with open('utils/max_sequence.txt', 'r') as file:
        max_sequence = int(file.read())
    return max_sequence

def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # This is the argument parser. I add the arguments below.
    ap.add_argument("-p",
                    "--prompt",
                    help="The prompt to start the text generation from.",
                    type = str)
    ap.add_argument("-w",
                    "--num_words",
                    help="The number of words to generate.",
                    type = int, default=10)
    args = ap.parse_args() # Parse the args
    return args

def load_saved_model():
    # Load tokenizer
    tokenizer = load(os.path.join(os.getcwd(), "models", "tokenizer.joblib"))
    
    # Define the maximum sequence length
    max_sequence_len = get_max_sequence_len()

    # Load the rnn model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "models", "rnn_model"))
    
    return tokenizer, max_sequence_len, model

def main():
    args = input_parser() # Parse the input arguments.
    tokenizer, max_sequence_len, model = load_saved_model()
    hf.generate_text(args.prompt, args.num_words, model, max_sequence_len, tokenizer)

if __name__ == "__main__":
    main()