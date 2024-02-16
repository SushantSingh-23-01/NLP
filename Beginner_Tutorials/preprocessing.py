import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


def preprocess(raw_string):
    #raw_string = raw_string.lower()
    text = re.sub(r'[^A-Za-z0-9 ]+',' ',raw_string) # remove characters except alpha numeric
    text = re.sub('\s+',' ',text)                   # remove whitespaces
    text = re.sub(r'\n','',text)                    # remove line breaks
    text = re.sub(r'\d+','',text)                   # remove digits
    
    bad_chars = ["●","•","|",'™','”','“']           # remove any further special char you encounter to the list  
    for char in bad_chars:
        text = text.replace(char,"")
    return text

def stopword_removal(raw_string):
    text = ' '.join([word for word in raw_string.split() if not word in set(stopwords.words('english'))])
    return text

def lemmatization(raw_string):
    list = []
    for token in word_tokenize(raw_string):
        list.append(WordNetLemmatizer().lemmatize(token))
    return ' '.join(list)

def stemming(raw_string):
    list = []
    for token in word_tokenize(raw_string):
        list.append(PorterStemmer().stem(token))
    return ' '.join(list)
