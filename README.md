BatchGPT
In this project, I have developed a code that takes in a dataset and analyzes it based on user prompts using LLMs like ChatGPT or Claude. The code outputs a dataframe.


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
ph.output_column=['index', aspect']
```

Running
```python
data=ph.run(df_cleaned)
df_aspects=pd.concat(ph.datas)
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
