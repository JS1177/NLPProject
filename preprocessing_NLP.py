#%%
import numpy as np
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import edit_distance as levenshteinDistance

from typing_extensions import Literal
#%%
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
#%%
VAR = {
    'data_path': os.path.join('UpdatedResumeDataSet_T1_7.csv'),
}
def clean_links(potentialLinks: list):
    
    '''
    Assumption: Potential link will always have at the minimum a .com
    
    Checks validity of link and returns cleaned link string
    '''
    
    assert isinstance(potentialLinks, list)
    
    http_exist = False
    www_exist = False
    com_exist = False
    
    if len(potentialLinks) < 1:
        return []
    
    ret_list = []
    
    for link in potentialLinks:
        
        http_match = re.search(r'(https?)(:)?(\/){0,2}', link)
        www_match = re.search(r'(www)(\.)?', link)
        com_match = re.search(r'(\.)?(com)', link)
        # print('flagged', link)
        
        #http
        if http_match != None:
            http_exist = True
        
        #www
        if www_match != None:
            www_exist = True
        
        #com
        if com_match != None:
            com_exist = True
            
        if (com_exist) or (com_exist and www_exist) or (com_exist and www_exist and http_exist):
            link = re.sub(r'(https?)(:)?(\/){0,2}', 'https://', link)
            link = re.sub(r'(www)(\.)?', 'www.', link)
            link = re.sub(r'(\.)?(com)', '.com', link)
            
            ret_list.append(link)
        else:
            #Not valid link
            ret_list.append(False)
            
    return ret_list
#%%
def clean_raw_text(text: str):
    
    # Clean links section
    potential_links = re.findall(
        r'(?:(?:https?:?\/\/{1,2})?w{1,3}\.?)?[a-zA-z0-9]{1,2048}\.?[a-zA-Z0-9]{1,6}\/\b[/\-a-zA-Z0-9]*\w', text
    ) 
    '''
    / will flag a sequence of characters as potential links
    
    Optional criteions: 
    http(s)
    //
    www & .
    . & com
    '''
    
    finalized_links = clean_links(potential_links)

    for potential_link, finalized_link in zip(potential_links, finalized_links):
        if finalized_link == False:
            continue
        else:
#             print('real_links', finalized_link)
            text = re.sub(potential_link, ' ', text) #Remove link
    
    #Clean non-characters
    text = re.sub(r'[^a-zA-Z0-9]', r' ', text)
    
    #Normalize text
    text = text.lower()

    #Clean whitespace section
    text = re.sub(r'[ ]{1,}', r' ', text)
    
    return text

# clean_raw_text(sample_res)
#%%
def check_numpy(text):
    
    if isinstance(text, list):
        text = np.array(text)
        return text
    elif isinstance(text, np.ndarray):
        return text
    else:
        raise TypeError('Not a list or numpy array')
#%%
def in_english_corpus(text: list | np.ndarray, behaviour: Literal['inside', 'outside'] = 'inside'):
    
    text = check_numpy(text)

    english_dictionary = nltk.corpus.words.raw().split('\n')

    english_dictionary = [word.lower() for word in english_dictionary] # normalize to lowercase
    
    word_in_dict_bool = np.isin(text, english_dictionary)
    
    if behaviour == 'inside':
        return text[word_in_dict_bool]
    elif behaviour == 'outside':
        word_not_in_dict_bool = np.invert(word_in_dict_bool)
        return text[word_not_in_dict_bool]
    else:
        return None
#%%
def clean_structured_text(text: list | np.ndarray, customer_dictionary: list = nltk.corpus.words.raw().split('\n')):
    
    #TODO There may be no point to cleaning mistyped random words > Intefere with keywords > Model may have to simply learn the noise

    text = check_numpy(text)
    
    customer_dictionary = [word.lower() for word in customer_dictionary] # normalize to lowercase
    
    word_in_dict_bool = np.isin(text, customer_dictionary)
    
    word_not_in_dict_bool = np.invert(word_in_dict_bool)
    
    
    
    words_in_dict = text[word_not_in_dict_bool]
    
    print(words_in_dict)
#%%
def wordnet_tag_format(tag: str):
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('A'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    
    return 'n' #Ensure lemmatize function can run
#%%
def extract_lemmas(tagged_tokens: list[tuple], lemmatizer=nltk.stem.WordNetLemmatizer()):
    lemmas = [lemmatizer.lemmatize(token[0], wordnet_tag_format(token[1])) for token in tagged_tokens]
    
    return lemmas
def extract_common_words_from_raw_data_ood(resumes_df: pd.DataFrame, column: str):
    
    resumes = resumes_df[column].to_numpy()
    resumes = check_numpy(resumes)

    lemmatizer = WordNetLemmatizer()
    
    counter = Counter()
    
    for index, resume in enumerate(resumes):
        normalized_resume = extract_lemmas(
            nltk.pos_tag(
                nltk.tokenize.word_tokenize(
                    clean_raw_text(resume))), 
            lemmatizer)
        
        # out of dictionary
        ood = in_english_corpus(normalized_resume, 'outside')
        counter.update(ood)
        
        if index % 10 == 0:
            print(index)
        
    return counter
def pipeline(filepath: str, feature_name: str):
    
    def total_normalize(text):
        
        text = clean_raw_text(text)
        
        # # Stopword Removal
        text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
        
        # Lemmatization
        text_tag = nltk.pos_tag(
            nltk.word_tokenize(text)
        )
        text_lemmas = extract_lemmas(text_tag)
        
        return ' '.join(text_lemmas)
    
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=[feature_name], keep='first')
    df[feature_name] = df[feature_name].apply(total_normalize)
    
    return df
