import string
import numpy as np
np.random.seed(420)

# keras module for building LSTMs
import tensorflow as tf
tf.random.set_seed(420)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Defines helper functions for the main program

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences, total_words):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    print("max_sequence_len: ", max_sequence_len)
    with open('utils/max_sequence.txt', 'w') as file:
        file.write(str(max_sequence_len))
    # make every sequence the length of the longest one by padding with 0s
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # add input embedding layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # add hidden layer 1 - LSTM layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # add output layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    print("PROMPT: ", seed_text)
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    
    return print("GENERATED TEXT: ", seed_text.title())

if __name__ == "__main__":
    pass