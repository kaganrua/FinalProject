from scripts import English_CNN, English_RNN
import pandas as pd
import os.path as osp


""""
WARNING: Although we've included scripts for both CNN for RNN codes, 
we highly recommend to run these experiments on a device with a GPU.

We will include the links to Jupyter notebooks for experiments with their(remark that there could be 
slight differences in implementation bc of different experiment environments(Google Collabs vs. MacOS))
Jupyter Notebook links: 
https://colab.research.google.com/drive/1AYmkv1UaMWYuFQuNMKYY5yzwsLdrf9nF?usp=sharing&fbclid=IwAR1XT_GPLGjC4wS2TqfXfaIx9H7aBU01xIFoLmSojb3dCL3VPqEq8TlqAaQ#scrollTo=5gBOBwnwlPn1
https://colab.research.google.com/drive/1JGdCBQtQ6l3GhLz00OMdkKGmI4sXL_Ry?usp=sharing&fbclid=IwAR10Hq30EsEdub8FZF_q7F40U_doKTuEunLgBOP6lHo0kw73bxUjmTYlOek#scrollTo=OldoablaDfjf
https://colab.research.google.com/drive/1SSbqNiwXCBu7v8b8waaozaSjh_RVliDf?usp=sharing&fbclid=IwAR3qpdUIUYQNEZiRYiWrjggAnn2fDF5dl5NGJDqPYs54JiNA0XgE6WxG3GM#scrollTo=TUFcmy3wG71r

"""

def main():
    overview()

    English_CNN.cnn(True, 'AG')
    English_CNN.cnn(False, 'AG')
    English_CNN(True, 'Yelp')
    English_CNN(False, 'Yelp')

    English_RNN.rnn(True , 'AG')
    English_RNN.rnn(False, 'AG')
    English_RNN.rnn(True, 'Yelp')
    English_RNN.rnn(False, 'Yelp')


def overview():
    YELP_TRAIN = pd.read_csv(osp.join('..' , 'data' , 'Raw_data' , 'english_data' , 'Yelp_train.csv'))
    YELP_TEST = pd.read_csv(osp.join('..', 'data', 'Raw_data', 'english_data', 'Yelp_test.csv'))

    AG_TRAIN = pd.read_csv(osp.join('..' , 'data' , 'Raw_data' , 'english_data' , 'train.csv') , names=['lasd' , 'Review'])
    AG_TEST = pd.read_csv(osp.join('..' , 'data' , 'Raw_data' , 'english_data' , 'test.csv'), names=['lasd' , 'Review'])

    print('Yelp has ' , len(YELP_TRAIN['Review']) , ' training instances and ' , len(YELP_TEST['Review']) , 'test instances')
    print('AG has ' , len(AG_TRAIN['Review']) , ' training instances and ' , len(AG_TEST['Review']))

if __name__ == '__main__':
    main()