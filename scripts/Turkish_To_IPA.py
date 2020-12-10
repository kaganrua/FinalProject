"""
Script for converting Turkish Data to IPA
"""
import pandas as pd
import os.path as osp
import epitran




def main():
    experiment = pd.read_csv(osp.join('..' , 'data' , 'Raw_data' , 'turkish_movie_sentiment_dataset.csv'))
    experiment['comment']  = experiment['comment'].str[23:]
    experiment['Label'] = 'NA'
    epi = epitran.Epitran('tur-Latn')



    for index ,row in experiment.iterrows():
        row['comment'] = epi.transliterate(row['comment'])
        if float(row['point'].replace(',' , '.')) >= 4.5:
            row['Label'] = 'Amazing'
        elif float(row['point'].replace(',' , '.')) >= 3.5:
            row['Label'] = 'Good'
        elif float(row['point'].replace(',' , '.')) >= 2.5:
            row['Label'] = 'Eh'
        elif float(row['point'].replace(',' , '.')) >=1.5:
            row['Label'] = 'Bad'
        else:
            row['Label'] = 'Awful'

    experiment.to_csv(osp.join('..' , 'data', 'IPA_Data', 'Turkish' , 'Turkish_IPA.csv'), index=False, encoding='utf-8')

    print('end of process')










if __name__ == '__main__':
    main()