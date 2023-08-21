import pandas as pd
import tiktoken
from tqdm import tqdm
from pandas.io import clipboard



class PromptHandler:
    def __init__(self):
        self.prompt=Prompt()
        self.counter=1
        self.start=0
        self.batches=[]
        
        
        self.LLM_function=Custom_LLM
    
    def run(self, df, merge_df: bool = True, max_tokens: int = 2000,
            row_max_tokens: int = 500, last_rows_margin_loss: int = 5):
        '''
        

        Parameters
        ----------
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

        Raises
        ------
        Exception
            Checking "id" and "text" columns.

        Returns
        -------
        list
            list of output DataFrames.

        '''
        
        #----Cheking columns
        allowed_cols={'id', 'text'}
        if len(set(allowed_cols).intersection(set(df.columns)))!=len(allowed_cols):
            raise Exception(f'df columns must be {allowed_cols}')
        
        #----droping long texts
        df=df.reset_index(drop=True)
        drop_idx=[]
        for i in tqdm(range(len(df))):
            if num_tokens_from_string(df.text[i])>row_max_tokens:
                drop_idx+=[df.index[i]]
        df=df.drop(drop_idx).reset_index(drop=True)
        print(len(drop_idx), 'droped due to long tokens')
        
        #----Processing prompts
        last_prompt=None
        last_token_count=0
        self.datas=[]
        while self.counter<len(df):  
            #batch is a variable that stores some rows of df
            batch=df[self.start: self.counter].reset_index(drop=True).reset_index()
            
            #temp_prompt a string that has prompt with data
            temp_prompt=self.prompt.glue(batch)
            
            '''Checking number of tokens, if it is ok prompt will be send, else 
            it will add more data to the temp_prompt
            '''
            token_count=num_tokens_from_string(temp_prompt)
            if token_count>max_tokens or self.counter==len(df)-1:
                
                #Final prompt
                self.prompt_text=last_prompt
                print('Token count: ', last_token_count, '|Rows:', self.counter, '/', len(df)-1)
                
                while True:
                    try:
                        #ouput of LLM
                        answer=self.LLM_function(self.prompt_text)
                        self.answer=answer
                        
                        #Processing output and converting to DataFrame
                        data=self.process_answer(answer)
                        self.batches+=[batch]
                        
                        #Merging output with input Dataframe
                        if merge_df:
                            data=batch.merge(data, on='index')
                            
                        self.datas+=[data]
                        self.counter-=1
                        self.start=self.counter
                        #Finishing Run function
                        if self.counter>len(df)-last_rows_margin_loss:
                            print('Done')
                            return self.datas
                        break
                    except Exception as e:
                        if str(e)=='Cooikes are expired!':
                            raise
                        print(e ,'trying again')
                        print('---answer---', answer)
                        # print(self.prompt_text)
                        # raise
          
                    
            else:
                last_prompt=temp_prompt
                last_token_count=token_count
            self.counter+=1
            
        
        return self.datas
    def process_answer(self, answer):
        '''
        Converting text output to DataFrame

        Parameters
        ----------
        answer : string
            DESCRIPTION.

        Returns
        -------
        data : DataFrame
            DESCRIPTION.

        '''
        data = pd.DataFrame([x.split(self.prompt.delimiter) for x in answer.split('\n')])
        data=data.rename(columns=data.iloc[0]).iloc[1:].reset_index(drop=True)
        data=data[data['index']!=''].reset_index(drop=True)
        data['index']=data['index'].astype('int')
        return data
        
        



class Prompt:
    '''
    This is an object for prompt things
    '''
    def __init__(self):
        #delimiter of output csv
        self.delimiter='|'
        
        #Information about your data
        self.info='here are some Instagram comments. the language is persian. '
        
        #Details about your tasks
        self.task='do aspect base sentiment analyses on this data. aspects must be in Persian. '
        
        #Columns of your output csv
        self.output_columns=['index', 'sentiment', 'aspect']
        
        #Warnings and tips to the LLM
        self.warns='dont do extra explanation.'

    
    def glue(self, batch):
        '''
        Glues all sub prompt and data together

        Parameters
        ----------
        batch : DataFrame
            DESCRIPTION.

        Returns
        -------
        string
            final prompt.

        '''
        self.output=f'items are seperated with |||. give the output in csv format with {self.delimiter} delimiter. mention columns name in first row. '
        data_prompt='\n'.join(['|||'+str(x[1]['index'])+'- '+x[1].text for c, x in enumerate(batch.iterrows())])
        # print('----------', data_prompt)
        columns=f' output columns are {", ".join(self.output_columns)}. '
        return self.info + self.task + columns + self.output + self.warns + '\n' + data_prompt
    

    
def num_tokens_from_string(string: str, encoding_name="gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def Custom_LLM(prompt):
    # print(prompt)
    addToClipBoard(prompt)
    print('Prompt is copied to the clipboard.')
    answer=input('Enter the output:')
    return answer
        


def addToClipBoard(text):
    # command = 'echo ' + text.strip() + '| clip'
    # os.system(command)
    clipboard.copy(text)

    

