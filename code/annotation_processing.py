import json
import pandas as pd
from pandas.io.json import json_normalize
import os

directory = '../data/annotation'

for filename in os.listdir(directory):
   if filename.endswith(".json"):
        filename_open = '../data/annotation/' + filename
        with open(filename_open) as json_file:
            data = json.load(json_file)


    #delete content and other information

            #j= len(data['cyberevent']['hopper'])

    #for event nugget

            # output1 = pd.DataFrame()
            # for i in range(0, j):
            #     output = data['cyberevent']['hopper'][i]['events']
            #     output2=json_normalize(output)
            #     output1=output1.append(output2)



    #for argument

            j= len(data['cyberevent']['hopper'])
    #
            output1 = pd.DataFrame()
            for i in range(0, j):
                output = data['cyberevent']['hopper'][i]['events']
                m = len(output)
                for l in range(0, m):
                    if 'argument' in output[l]:
                        output2 = json_normalize(output[l], record_path=['argument'])
                        output1 = output1.append(output2)
    #
    # print(output1)


            filename = '../data/argument_annotation/' + filename + '.csv'
            with open(filename,'w') as outfile:
                output1.to_csv(filename, index=None,
                                header=True)

        continue
   else:
        continue


