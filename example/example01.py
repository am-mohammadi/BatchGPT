import pandas as pd
from BatchGPT import Cleaner
from BatchGPT.Prompt import PromptHandler


df=pd.DataFrame([{'id': 0, 'text': 'i feel good today'},
                 {'id': 1, 'text': 'i feel bad today'},
                 {'id': 2, 'text': 'i am fine'},
                 {'id': 3, 'text': 'i feel good today'},
                {'id': 4, 'text': 'i feel bad today'},
                {'id': 5, 'text': 'i am fine'},
                {'id': 6, 'text': 'i feel good today'},
                {'id': 7, 'text': 'i feel bad today'},
                {'id': 8, 'text': 'i am fine'},
                 
    ])

# df
#    id               text
# 0   0  i feel good today
# 1   1   i feel bad today
# 2   2          i am fine
# 3   3  i feel good today
# 4   4   i feel bad today
# 5   5          i am fine
# 6   6  i feel good today
# 7   7   i feel bad today
# 8   8          i am fine




df_cleaned=Cleaner.clean(df.copy())

ph=PromptHandler()

ph.prompt.delimiter=','

ph.prompt.info='here are some comments. '
ph.prompt.task='do sentiment analysis on this data.'
ph.prompt.output_columns=['index', 'sentiment']

data=ph.run(df_cleaned, merge_df = True, max_tokens = 2000, row_max_tokens = 400, last_rows_margin_loss=5)

df_sentiments=pd.concat(data)
df_sentiments.to_csv('sentimentss.csv', index=False , encoding='utf-8-sig')

# df_sentiments
# index,sentiment
# 0,positive
# 1,negative
# 2,neutral
# 3,positive
# 4,negative
# 5,neutral
# 6,positive
# 7,negative

