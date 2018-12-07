# pandas and numpy packages
import pandas as pd
from pandas import get_dummies
import numpy as np

# sklearn package and required libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler , Normalizer
from sklearn.base import BaseEstimator, TransformerMixin

# DenseTransformer from mlxtend package
from mlxtend.preprocessing import DenseTransformer

# plot visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

# nltk package
from nltk.stem.snowball import SnowballStemmer
import nltk

# others
import re
import calendar

# disable Deprecation Warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# loading the data into dataframe - please provide the path to .tsv file 
dr_data = pd.read_table('drugsDataRaw.tsv')

# add review lenght (characters count) as a new feature
dr_data['reviewLength'] = dr_data['review'].apply(len)

# add review lenght (words count) as a new feature
dr_data['reviewWords'] = dr_data['review'].str.lower().str.split().apply(len)

# to better investigate/use the date feature, we split it to day, month, and year features
# we first create a dictionary to map month name to month number
month_dict = {v: k for k,v in enumerate(calendar.month_name)}

dr_data['day']= dr_data['date'].map(lambda x: int(x.split(',')[0].split()[1]))
dr_data['month']= dr_data['date'].map(lambda x: month_dict[x.split(',')[0].split()[0]])
dr_data['year']= dr_data['date'].map(lambda x: int(x.split(',')[1]))

# we define the text_process function to clean and pre-process the text features

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")

# here we define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def text_process(text,stem=False,rm_stopwords=True):
    output = []
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    output = filtered_tokens
    if stem:
        output = [stemmer.stem(t) for t in filtered_tokens]
    if rm_stopwords:
        output = [t for t in output if t.lower() not in stopwords]
    return output


# we process condition feature and keep the processed text in a new column 'conditionProcessed'
dr_data['conditionProcessed'] = dr_data.apply (lambda row: ' '.join(text_process(str(row['condition']))),axis=1)

# we process the reviews and keep the processed text in a new column 'reviewProcessed'
dr_data['reviewProcessed'] = dr_data.apply (lambda row: ' '.join(text_process(str(row['review']),stem=False)),axis=1)

# splitting features and labels for classification. we load the features into 'X' and the label (rating) into 'Y'. 
X = dr_data[['conditionProcessed','reviewProcessed','reviewLength','usefulCount','day','month','year']]
Y = np.array(dr_data['rating'])
# splitting the data into training and test sets. We keep %30 of the data for test. 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# We use Sklearn's pipeline functionality to easily repeat commonly occuring 
# steps (e.g., feature extraction, feature transformation, learning) in our modeling process.

# First thing we want to do is define how to process our features. 
# The standard preprocessing in pipelines apply the same preprocessing to the whole dataset, 
# but in cases where we have heterogeneous data, this doesn't quite work. So first we create 
# some selectors/transformers that simply returns the one column in the dataset by the key value we pass.

# We define two different selectors for either text or numeric columns.

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

# We also define the following transformers/vectorizers to be used for one hot encoding and word embedding.

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    
# We will implement an embedding vectorizer - a counterpart of CountVectorizer and TfidfVectorizer - 
# that is given a word -> vector mapping and vectorizes texts by taking the mean of all the vectors 
# corresponding to individual words.
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

# For each feature, we make a mini pipeline that consists of two steps: 
# 1. grab just that column from the dataset
# 2. perform proper transform on just that column and return the results.


# For review feature, we apply tf-idf vectorizer with l2-normalization and using unigram and bigrams
# (the parameters has been tuned through grid search)
review = Pipeline([
            ('review_extract' , TextSelector(key='reviewProcessed')),
            ('tfidf', TfidfVectorizer(max_features=100000,
                             stop_words='english',
                             norm='l2',use_idf=True, ngram_range=(1,2)))
        ])

# For condition feature, we apply tf-idf vectorizer without normalization and using unigram 
condition = Pipeline([
            ('cond_extract', TextSelector(key='conditionProcessed')),
            ('tfidf', TfidfVectorizer(max_features=100000,
                             stop_words='english',
                             norm=None,use_idf=True, ngram_range=(1,1)))
        ])

# For review length and usefulCount, we apply simple numerical normalizer 
reviewLen = Pipeline([
            ('reviewLength', NumberSelector(key='reviewLength')),
            ('scale', Normalizer())
        ])

useful = Pipeline([
            ('usefulCount_extract', NumberSelector(key='usefulCount')),
            ('scale', Normalizer())
        ])

# We consider day, month, and year as categorical features, and apply on-hot encodding followed by dense transformer
day = Pipeline([
            ('day_extract', NumberSelector(key='day')),
            ('one_hot', OneHotEncoder()),
            ('to_dense', DenseTransformer())
        ])

month = Pipeline([
            ('month_extract', NumberSelector(key='month')),
            ('one_hot', OneHotEncoder()),
            ('to_dense', DenseTransformer())
        ])

year = Pipeline([
            ('year_extract', NumberSelector(key='year')),
            ('one_hot', OneHotEncoder()),
            ('to_dense', DenseTransformer())
        ])


# To make a pipeline from all of our pipelines, we do the same thing, but now we 
# use a FeatureUnion to join the feature processing pipelines.
# Using a FeatureUnion, we actually parallelize the processesing/transforming our features. 
featsUnion = FeatureUnion([('review', review),
                      ('condition',condition),
                      #('reviewLen',reviewLen), # uncomment this line to include reviewLen in features
                      ('useful',useful),
                      #('day',day), # uncomment this line to include reviewLen in features
                      #('month',month), # uncomment this line to include reviewLen in features
                      #('year',year), # uncomment this line to include reviewLen in features
                     ])

# Finally, we create a classifier pipeline to generate predictions based on the given features.
classifierPipeline = Pipeline([
    ('features', featsUnion),
    ('classifier', LogisticRegressionCV(cv=5))
])

# We have also evaluated the following classifiers: 
# ExtraTreesClassifier(n_estimators=200)
# SVC(kernel="linear")
# LogisticRegressionCV(cv=5)
# RandomForestClassifier(random_state=0, n_estimators=70)
# SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
# however, LogisticRegressionCV (logistic regression with cross validation) achievd the best results.


# run the classifier pipeline and print the result
classifierPipeline = classifierPipeline.fit(X_train, Y_train)
predictions = classifierPipeline.predict(X_test)
# calculating Accuracy, MAE, MSE, RMSE
print('Accuracy: ',np.mean(predictions == Y_test))
print('MAE:',metrics.mean_absolute_error(Y_test, predictions))
print('MSE:',metrics.mean_squared_error(Y_test, predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test, predictions)))

# print the confusion matrix and report
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, predictions))
print('\n')
print(classification_report(Y_test, predictions))

