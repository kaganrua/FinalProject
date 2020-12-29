import math
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import os.path as osp
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Activation, Flatten, Dense, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


# from google.colab import drive
# drive.mount('/content/drive')




   # rnn(True, 'Yelp')
    #rnn(False, 'Yelp')
  #  rnn(True, 'AG')
  #  rnn(False, 'Yelp')




def rnn(IPA, Corpus):
    if IPA == False and Corpus == 'AG':
        TEST = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'test.csv'),
                       index_col=False, names=["Encoded", "Title", "DESCRIPTION"])

        TEST['Review'] = TEST['Title'] + ' ' + TEST['DESCRIPTION']

        TRAIN = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'train.csv'),
                        index_col=False, names=["Encoded", "Title", "DESCRIPTION"])

        TRAIN['Review'] = TRAIN['Title'] + TRAIN['DESCRIPTION']
    elif IPA == True and Corpus == 'AG':
        TEST = pd.read_csv(osp.join('..', 'data', 'IPA_data', 'English', 'AG_IPA_test.csv'))
        TRAIN = pd.read_csv(osp.join('..', 'data', 'IPA_data', 'English', 'AG_IPA_train.csv'))
    elif IPA == False and Corpus == 'Yelp':
        TEST = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'Yelp_test.csv'))
        TRAIN = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'Yelp_test.csv'))
    else:
        TEST = pd.read_csv(osp.join('..', 'data', 'IPA_data', 'English', 'Yelp_IPA_test.csv'))
        TRAIN = pd.read_csv(osp.join('..', 'data', 'IPA_data', 'English', 'Yelp_IPA_train.csv'))

    print(TRAIN['Encoded'].unique())

    texts_train = TRAIN['Review'].values
    # problems with encoding in Yelp_IPA
    if IPA == True and Corpus == 'Yelp':
        texts_train = [str(s) for s in texts_train]
    texts_train = [s.lower() for s in texts_train]

    texts_test = TEST['Review'].values
    texts_test = [s.lower() for s in texts_test]


    print(len(texts_train))


    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')


    alphabet="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = dict()
    for i, char in enumerate(alphabet):
      char_dict[char] = i + 1

    tk.word_index = char_dict
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
    print('Vocabulary:\n' ,tk.word_index)



    train_sequences = tk.texts_to_sequences(texts_train)
    test_sequences = tk.texts_to_sequences(texts_test)




    train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
    test_data = pad_sequences(test_sequences, maxlen=1014, padding='post')

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # problems with encoding in Yelp_IPA



    train_class_list = TRAIN['Encoded'].values
    train_class_list = [x-1 for x in train_class_list]

#TEST['Encoded'] = np.where(TEST['Label'].str.contains("positive"), 1 , 0)
    test_class_list = TEST['Encoded'].values
    test_class_list = [x-1 for x in test_class_list]

    from keras.utils import to_categorical

    train_classes = to_categorical(train_class_list)
    test_classes = to_categorical(test_class_list)

    vocab_size = len(tk.word_index)

    embedding_weights = []

    embedding_weights.append(np.zeros(vocab_size)) #Zero vector for representing PAD

    for char, i in tk.word_index.items():
        onehot = np.zeros(vocab_size)
        onehot[i-1] = 1
        embedding_weights.append(onehot)


    input_size = 1014

    embedding_size = vocab_size

    num_of_classes = len(TRAIN['Encoded'].unique())

    embedding_weights = np.array(embedding_weights)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    X_train = train_data[indices]
    Y_train = train_classes[indices]

    X_test = test_data
    Y_test = test_classes

    filepath = "model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]


# model
# model = Sequential()
# model.add(Embedding(vocab_size+1, embedding_size, input_length=input_size, weights=[embedding_weights]))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(num_of_classes, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(train_data, train_classes, validation_data=(X_test, Y_test), epochs=3, batch_size=128, verbose=1)

# model 2
# Input for variable-length sequences of integers
    inputs = keras.Input(shape=(input_size,), dtype="int64")
# Embed each integer in a 128-dimensional vector
    x = layers.Embedding(vocab_size+1, embedding_size, input_length=input_size)(inputs)
# Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
    outputs = layers.Dense(num_of_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()


# define the checkpoint
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    summary = model.fit(train_data, train_classes, batch_size=128, epochs=20, validation_data=(X_test, Y_test), verbose=1, callbacks=callbacks_list)



    plt.plot(summary.history['accuracy'])
    plt.plot(summary.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train' ,'Test'] , loc='upper left')
    plt.show()

    plt.plot(summary.history['loss'])
    plt.plot(summary.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()











    print('DEBUG')


