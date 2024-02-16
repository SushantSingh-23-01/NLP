import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# !kaggle datasets download -d yasserh/imdb-movie-ratings-sentiment-analysis
path = r'datasets/movie.csv'
split_size = 0.2

def preprocess(raw_text):
    text = re.sub(r'[^A-Za-z0-9 ]+',' ',raw_text)
    text = re.sub('\s+',' ',text)
    text = re.sub(r'\n','',text)
    text = re.sub(r'\d+','',text) 
    return text

def load_dataset(path):
    dataset = pd.read_csv(path)
    x = dataset['text'].apply(preprocess)
    y = dataset['label']
    return x,y

def split(x,y,split_size):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size=split_size,random_state=42)
    return x_train, x_test, y_train, y_test 

def vectorizer(x_train,x_test):
    vectorizer = CountVectorizer(stop_words = 'english')
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return x_train_vec, x_test_vec

def logreg(x_train_vec,y_train,x_test_vec,y_test):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train_vec,y_train)
    y_pred = classifier.predict(x_test_vec)
    score = classifier.score(x_test_vec,y_test)
    return score,y_pred

def naivebayes(x_train_vec,y_train,x_test_vec,y_test):
    classifier = MultinomialNB()
    classifier.fit(x_train_vec,y_train)
    y_pred = classifier.predict(x_test_vec)
    score = classifier.score(x_test_vec,y_test)
    return score,y_pred

def svd(x_train_vec,y_train,x_test_vec,y_test):
    classifier = LinearSVC(dual='auto')
    classifier.fit(x_train_vec,y_train)
    y_pred = classifier.predict(x_test_vec)
    score = classifier.score(x_test_vec,y_test)
    return score,y_pred

def plot_cm(y_test,y_pred):    
    cm = confusion_matrix(y_true=y_test,y_pred=y_pred,labels=None)
    df_cm = pd.DataFrame(cm,index = ['Positive','Negative'], columns = ['Positive','Negative'])
    return df_cm

def main():
    x,y = load_dataset(path)
    x_train, x_test, y_train, y_test = split(x,y,split_size)
    x_train_vec, x_test_vec = vectorizer(x_train,x_test)
    score,y_pred = logreg(x_train_vec,y_train,x_test_vec,y_test)
    print(f'Logistic Regression Model : Accuracy : {score}')
    score,y_pred = naivebayes(x_train_vec,y_train,x_test_vec,y_test)
    print(f'\nMultinomial Naive Bayes Model : Accuracy : {score}')
    score,y_pred = svd(x_train_vec,y_train,x_test_vec,y_test)
    print(f'\nSupport Vector Machine Model : Accuracy : {score}')
    
if __name__ == '__main__':
    main()
