BatchGPT
In this project, I have developed a code that takes in a dataset and analyzes it based on user prompts using LLMs like ChatGPT or Claude. The code outputs a dataframe.

A simple daigram
![BatchGPT drawio (1)](https://github.com/am-mohammadi/BatchGPT/assets/60970821/478f4648-d922-4ca1-ad8b-fc4db8a7f220)


Basic details you should set
```python
PromptHandler.prompt.info='your info about the dataset'
PromptHandler.prompt.task='your task details'
PromptHandler.prompt.output_columns=['index', 'your other columns']
```
----------------------------------------------
Example for Instagram comments:

Cleaning Dataset
```python
from BatchGPT import Cleaner
import pandas as pd
#Dataset Must contain 'id' and 'text' columns
df=pd.read_csv('data.csv', usecols=['pk', 'text']).rename(
    columns={'pk': 'id'})
#Cleaning text
df_cleaned=Cleaner.clean(df.copy())
#it's optional to translate
#df_cleaned=Cleaner.translate(df_cleaned)
```

Writing Prompts
```python
from BatchGPT.Prompt import PromptHandler
#Loading Cleaned dataset
ph=PromptHandler()
ph.prompt.info='here are some Instagram comments about a post. the post is a video that shows blah blah. "Mohsen F" is the owner of the post.'
ph.prompt.task='tell from what aspect each comment criticizes Mohsen F. aspects title must be short.'
ph.prompt.output_columns=['index', aspect']
```

Running
```python
data=ph.run(df_cleaned)
'''
df : DataFrame
            DataFrame with "id" and "text" columns.
        merge_df : bool, optional
            Set this True if your output will merge to the input by "id". The default is True.
        max_tokens : int, optional
            Maximum number of token for each prompt. The default is 2000.
        row_max_tokens : int, optional
            Maximum number of token for each row of text. if text has more tokens it will removed. The default is 500.
        last_rows_margin_loss : int, optional
            Number of rows that will be ignored in the last prompt in case of error. The default is 5.
'''
df_aspects=pd.concat(data)
df_aspects.to_csv('aspects.csv', index=False , encoding='utf-8-sig')

```

Default LLM_function is Custom_LLM that will copy to clipboard the prompt and 
you need to paste in your LLM chat like claude or GPT and copy the answer 
to the code.
also you can assign your api function from LLM that its input is prompt and 
the output is answer text

```python
PromptHandler.LLM_function=YOUR_LLM_FUNCTION
```

Example
```python
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


```


