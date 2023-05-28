import os 
import pandas as pd
import argparse
from joblib import dump


import tensorflow as tf
tf.random.set_seed(420)
from tensorflow.keras.preprocessing.text import Tokenizer

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# import helper functions
import sys
sys.path.append(os.path.join(os.getcwd()))
import utils.helper_functions as hf

def input_parser(): # This function parses the input arguments and returns an object containing the arguments.
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--epochs",
                    help="Amount of epochs to train the model for.",
                    type = int, default=89)
    ap.add_argument("--batch_size",
                    help="Batch size to use when training the model.",
                    type = int, default=64)
    ap.add_argument("--all_files", help = "Select this as true if you want to load all files from your directory.",
                    type = bool, default = False),
    ap.add_argument("--common_word", help = "The common word in the file names, if you are loading selected files.",
                    type = str, default = "Comments")
    ap.add_argument("--column_name", help = "The column name of the text entries in the csv files.", 
                    type = str, default = "commentBody")
    args = ap.parse_args() # Parse the args
    return args

# set the data directory
def get_data_dir():
    data_dir = os.path.join(os.getcwd(), "data")
    
    return data_dir

# extract data from multiple csv files
def load_comments(data_dir, all_files, common_word, column_name):
    all_comments = []
    for filename in os.listdir(data_dir):
        if not all_files:
            if common_word in filename:
                comment_df = pd.read_csv(os.path.join(data_dir, filename))
                all_comments.extend(list(comment_df[column_name].values))
        else:
            # just get all .csv files from the directory with the column name
            comment_df = pd.read_csv(os.path.join(data_dir, filename))
            all_comments.extend(list(comment_df[column_name].values))
    return all_comments

# remove unknown comments
def preprocess_comments(all_comments):
    #all_comments = all_comments[:1000] # use only the first 1000 comments for testing
    all_comments = [c for c in all_comments if c != "Unknown"]
    # define the corpus and clean it
    corpus = [hf.clean_text(x) for x in all_comments]

    return corpus

def tokenize_corpus(corpus):
    # load the tokenizer and fit on the corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # define the input sequences
    inp_sequences = hf.get_sequence_of_tokens(tokenizer, corpus)

    # generate the padded sequences
    predictors, label, max_sequence_len = hf.generate_padded_sequences(inp_sequences, total_words)
    
    return predictors, label, max_sequence_len, total_words, tokenizer

def train_model(epochs_arg, batch_size_arg, predictors, label, max_sequence_len, total_words):
    # intialize the model
    model = hf.create_model(max_sequence_len, total_words)
    model.summary()

    # train the model
    model.fit(predictors, 
                label, 
                epochs=int(epochs_arg),
                batch_size=int(batch_size_arg), 
                verbose=1)
    
    return model

def save_models(model, tokenizer):
    # save the model
    model.save(os.path.join(os.getcwd(), "models", "rnn_model"))
    # save the tokenizer with joblib
    dump(tokenizer, os.path.join(os.getcwd(), "models", "tokenizer.joblib"))

def main():
    args = input_parser() # Parse the input arguments.
    data_dir = get_data_dir() # Get the data directory.
    print("Loading comments...")
    all_comments = load_comments(data_dir, args.all_files, args.common_word, args.column_name) # Load the text data.
    print("Preprocessing comments...")
    corpus = preprocess_comments(all_comments) # Preprocess the text data.
    print("Tokenizing corpus...")
    predictors, label, max_sequence_len, total_words, tokenizer = tokenize_corpus(corpus) # Tokenize the corpus.
    print("Training model...")
    model = train_model(args.epochs, args.batch_size, predictors, label, max_sequence_len, total_words) # Train the model.
    save_models(model, tokenizer) # Save the model and tokenizer.
    print("Done!")
    
if __name__ == "__main__":
    main()