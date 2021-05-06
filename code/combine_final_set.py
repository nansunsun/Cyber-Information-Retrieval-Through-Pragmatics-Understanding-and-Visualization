##combine all csv files to a single csv

### one way to combine csv
# os.chdir("../data/text_label_nugget")
# extension = 'csv'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# combined_csv = combined_csv[['sentences.index','originalText','tag']]
# combined_csv.columns=['Sentence #', 'Word', 'Tag']
# combined_csv['Tag']= combined_csv['Tag'].replace({'0':'O'})

##another way with the filename
import os
import glob
import pandas as pd

path = "../data/labelled argument with pos ner"
all_files = glob.glob(os.path.join(path, "*.csv"))
names = [os.path.basename(x) for x in glob.glob(path+'\*.csv')]
df = pd.DataFrame()

for file_ in all_files:
    file_df = pd.read_csv(file_, index_col=0, header=0)
    file_df['file_name'] = file_
    df = df.append(file_df)

df = df[['sentences.index','originalText','tag','pos','ner','lemma','file_name']]
df.columns=['Sentence #', 'Word','Tag','pos','ner','lemma','file_name']
df['Tag']= df['Tag'].replace({'0':' O'})

df['group id'] = df.groupby(['Sentence #','file_name']).ngroup()
#

df = df[['group id','Word','Tag','pos','ner','lemma']]
df.columns=['Sentence #', 'Word', 'Tag','PoS','NER','Lemma']

#
df.to_csv("argument_label_summary_withpos.csv", index = False)