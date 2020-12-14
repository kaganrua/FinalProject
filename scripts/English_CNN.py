import math
import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import os.path as osp
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Activation, Flatten, Dense, Conv1D, MaxPooling1D, Dropout
import matplotlib.pyplot as plt



def main():
    cnn(True)

def cnn(IPA):
    if IPA==False:
        TEST = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'test.csv'),
                            index_col=False , names=["Encoded" , "Title" , "DESCRIPTION"])

        TEST['Review'] = TEST['Title'] + TEST['DESCRIPTION']

        TRAIN = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'train.csv'),
                             index_col=False, names=["Encoded" , "Title" , "DESCRIPTION"])

        TRAIN['Review'] = TRAIN['Title'] + TRAIN['DESCRIPTION']
    else:
        TEST = pd.read_csv(osp.join('..', 'data', 'IPA_data', 'English', 'AG_IPA_test.csv'))
        TRAIN = pd.read_csv(osp.join('..', 'data', 'IPA_data', 'English', 'AG_IPA_train.csv'))
    #print(TEST)
    #print(TRAIN)


    print(TRAIN['Encoded'].unique())
    texts_train = TRAIN['Review'].values
    texts_train = [s.lower() for s in texts_train]

    texts_test = TEST['Review'].values
    texts_test = [s.lower() for s in texts_test]


    print(len(texts_train))


    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

    if IPA == False:
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        char_dict = dict()
        for i, char in enumerate(alphabet):
            char_dict[char] = i + 1

        tk.word_index = char_dict
        tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
        print('Vocabulary:\n' ,tk.word_index)
    else:
        tk.fit_on_texts(texts_train)
        print('Vocabulary:\n' ,tk.word_index)



    train_sequences = tk.texts_to_sequences(texts_train)
    test_sequences = tk.texts_to_sequences(texts_test)




    train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
    test_data = pad_sequences(test_sequences, maxlen=1014, padding='post')

    train_data = np.array(train_data)
    test_data = np.array(test_data)



    #TRAIN['Encoded'] = np.where(TRAIN['Label'].str.contains("positive"), 1 , 0)
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

    conv_layers = [[256,7,3],
                   [256,7,3],
                   [256,3,-1],
                   [256,3,-1],
                   [256,3,-1],
                   [256,3,3]]

    fully_connected_layers = [1024,1024]
    num_of_classes = 4
    dropput_p = 0.5
    optimizer = 'adam'
    loss_function = 'categorical_crossentropy'



    embedding_weights = np.array(embedding_weights)

    #initialize embedding layer
    embedding_layer = Embedding(vocab_size+1, embedding_size, input_length=input_size, weights=[embedding_weights])

    #Input Layer

    inputs = Input(shape=(input_size,) , name='input', dtype='int64')

    x = embedding_layer(inputs)

    for filter_num, filter_size, pooling_size in conv_layers:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation('relu')(x)
        if pooling_size != -1:
            x = MaxPooling1D(pool_size=pooling_size)(x)

    x = Flatten()(x)

    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation='relu')(x)
        x = Dropout(dropput_p)(x)

    predictions = Dense(num_of_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    print(model.summary())

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    X_train = train_data[indices]
    Y_train = train_classes[indices]

    X_test = test_data
    Y_test = test_classes

    summary = model.fit(train_data, train_classes, validation_data=(X_test, Y_test), batch_size=128, epochs=10 , verbose=2)


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



if __name__ == '__main__':
    main()