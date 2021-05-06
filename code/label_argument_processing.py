import pandas as pd
import os

directory_text = '../data/source_text_csv'
directory_anno = '../data/argument_annotation'

for filename in os.listdir(directory_text):
    if filename.endswith(".txt.csv"):
        filename = filename[:-7]
        text = pd.read_csv('../data/source_text_csv/'+ filename + 'txt.csv')
        annotation = pd.read_csv('../data/argument_annotation/' + filename + 'json.csv')

#text=pd.read_csv('../data/source_text_csv/5.txt.csv')
#annotation =pd.read_csv('../data/nugget_annotation/5.json.csv')

    text['tag'] = 0

    text_row_number = len(text)
    annotation_row_number = len(annotation)

    for i in range (0, annotation_row_number):
        for j in range (0, text_row_number):
            anno = annotation['startOffset'][i]
            tex  = text['characterOffsetBegin'][j]
            if anno == tex:
                print(i,j)
                tag = annotation['type'][i]
                nugget_text = annotation['text'][i]
                nugget_len = nugget_text.count(' ') + 1
                print(nugget_len)
                if nugget_len == 1:
                    text.loc[j,'tag'] = 'B-' + tag
                else:
                    text.loc[j,'tag'] = 'B-' + tag
                    for k in range(1,nugget_len):
                        text.loc[j+k, 'tag'] = 'I-' + tag
    print(text)
    text.to_csv(filename + 'csv')



            #

    #     continue
    # else:
    #     continue
    #





















