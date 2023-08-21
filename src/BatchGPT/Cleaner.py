import re
from deep_translator import GoogleTranslator
import time
import pandas as pd
from tqdm import tqdm


def extract_hashtag(text):
        return re.findall(r"#(\w+)", text)
    
def translate(texts, source='fa', target='en'):
        '''
        

        Parameters
        ----------
        texts :  pandas dataframe
            DESCRIPTION.
        source : str, optional
            DESCRIPTION. The default is 'fa'.
        target : str, optional
            DESCRIPTION. The default is 'en'.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        pandas dataframe
            translated texts

        '''
        if type(texts)!=pd.core.frame.DataFrame:
            raise Exception('texts must be pandas dataframe!')
        if len(set(texts.columns).intersection(set(['id', 'text'])))!=2:
            raise Exception('texts dataframe must have id & text columns!')
            
        print('translating texts...')
        translated_texts=[]
        for i in tqdm(range(len(texts))):
            text=texts.text[i]
            while True:
                try:
                    translated = GoogleTranslator(source=source, target=target).translate(text)
                    break
                except:
                    print('failed to translate, trying again...')
                    time.sleep(2)
            translated_texts.append({'id': texts.id[i],
                                          'text': translated
                                          })
        translated_texts=pd.DataFrame(translated_texts)
        translated_texts=translated_texts.dropna().reset_index(drop=True)
        return translated_texts
    
def clean(texts, remove_mention=False):
    texts['text']=texts['text'].map(lambda x: text_cleaner(x, remove_mention=remove_mention))
    texts=texts.dropna().reset_index(drop=True)
    print('texts cleaned.')
    return texts
    
def text_cleaner(text, remove_mention=False):
    text=str(text)
    temp=[]
    if '@' in text and remove_mention:
        return None
    for token in text.split(' '):
        if '@' not in token:
            temp+=[token]
    text=' '.join(temp)
    
    text = re.sub("@[A-Za-z0-9]+","",text) #Remove @ sign
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) #Remove http links
    text=re.sub('[()!?=]', ' ', text)
    text = " ".join(text.split())
    # text = ''.join(c for c in text if not emoji.is_emoji(c)) #Remove Emojis
    text = text.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    # text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
         # if w.lower() in words or not w.isalpha())
    if len(text)==0:
        return None
    return text


