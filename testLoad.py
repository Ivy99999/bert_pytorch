
# -*- coding:utf-8 -*-

import pickle
import pandas as pd
df_path="./output_wiki_bert/df_log.pickle"
# obj2 = pickle.loads(df_path)
# print(obj2)
df = pd.read_pickle(df_path)
print(df)
# with open(df_path, 'r',encoding='utf-8') as f:
#     data = pickle.load(f)
#     print(data)