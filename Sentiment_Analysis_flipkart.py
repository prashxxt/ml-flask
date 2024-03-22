#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("P:\\INNMOATICS\\task ml\\reviews_badminton\\data.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.loc[0, 'Review text']


# In[8]:


df["Review Title"]


# In[9]:


df.isnull().sum()


# In[10]:


data = df[['Review text', 'Ratings']]   # Selecting only the relevant columns


# In[11]:


data.columns = ['review_text', 'sentiment']  # Renaming columns for convenience


# In[12]:


data


# In[13]:


data.dtypes


# In[14]:


data.describe()


# In[15]:


data.dropna(subset=['review_text'], inplace=True)

# Step 3: Reset index (optional)
data.reset_index(drop=True, inplace=True)


# In[16]:


data.isnull().sum()


# In[17]:


data['review_text']


# #Data Preparation

# In[18]:


data


# In[19]:


data["sentiment"].value_counts(normalize = True)


# In[20]:


X=data[['review_text']]
y=data[['sentiment']]


# In[21]:


# Splitting into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


X_train.info()


# In[23]:


print(X_train.head())


# In[24]:


X_train.shape


# # Text Preprocessing

# In[25]:


import nltk

# Download the punctuations
nltk.download('punkt')
# Download the stop words corpus
nltk.download('stopwords')
# Downloading wordnet before applying Lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[26]:


import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[27]:


stemmer = PorterStemmer()


# In[28]:


lemmatizer = WordNetLemmatizer()


# In[29]:


def preprocess(raw_text, flag):
    # Removing special characters and digits
    text = re.sub("[^a-zA-Z]", " ", raw_text)

    # change text to lower case
    text = text.lower()

    # tokenize into words
    words = word_tokenize(text)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # Stemming/Lemmatization
    if(flag == 'stem'):
        words = [stemmer.stem(word) for word in words]
    else:
        words = [lemmatizer.lemmatize(word) for word in words]

    preprocessed_text = " ".join(words)
    words_in_preprocessed_text = len(words)

    return pd.Series([preprocessed_text, words_in_preprocessed_text])


# In[30]:


temp_df=X_train['review_text'].apply(lambda x: preprocess(x, 'stem'))
temp_df


# In[31]:


from tqdm import tqdm, tqdm_notebook


# In[32]:


tqdm.pandas()


# In[33]:


temp_df = X_train['review_text'].progress_apply(lambda x: preprocess(x, 'stem'))

temp_df.head()


# In[34]:


temp_df.columns = ['clean_text_stem', 'text_length_stem']

temp_df.head()


# In[35]:


X_train = pd.concat([X_train, temp_df], axis=1)

X_train.head()


# In[36]:


temp_df = X_train['review_text'].progress_apply(lambda x: preprocess(x, 'lemma'))

temp_df.head()


# In[37]:


temp_df.columns = ['clean_text_lemma', 'text_length_lemma']

temp_df.head()


# In[38]:


X_train = pd.concat([X_train, temp_df], axis=1)

X_train.head()


# In[39]:


X_train.head()


# Creating Word Cloud

# In[40]:


from wordcloud import WordCloud


# In[41]:


y_train


# In[42]:


y_train_df = pd.DataFrame(y_train, columns=['sentiment'])
y_train_df


# In[43]:


positive_indices = y_train_df[(y_train_df['sentiment'] == 5) | (y_train_df['sentiment'] == 4)].index
positive_df = X_train.loc[positive_indices]


# In[44]:


positive_df


# In[45]:


words = ' '.join(positive_df['clean_text_lemma'])

print(words[:100])


# In[46]:


cleaned_word = " ".join([word for word in words.split()
                        if 'subject' not in word])


# In[47]:


positive_wordcloud = WordCloud(stopwords=stopwords.words("english"),
                      background_color='black',
                      width=1600,
                      height=800
                     ).generate(cleaned_word)


# In[48]:


import matplotlib.pyplot as plt


# In[49]:


plt.figure(1,figsize=(30,20))
plt.imshow(positive_wordcloud)
plt.axis('off')
plt.show()


# In[50]:


negative_indices = y_train_df[(y_train_df['sentiment'] == 1) | (y_train_df['sentiment'] == 2)].index
negative_df = X_train.loc[negative_indices]


# In[51]:


negative_df


# In[52]:


words = ' '.join(negative_df['clean_text_lemma'])

cleaned_word = " ".join([word for word in words.split()
                        if 'subject' not in word])


# In[53]:


neg_wordcloud = WordCloud(stopwords=stopwords.words("english"),
                      background_color='black',
                      width=1600,
                      height=800
                     ).generate(cleaned_word)


# In[54]:


plt.figure(1,figsize=(30,20))
plt.imshow(neg_wordcloud)
plt.axis('off')
plt.show()


# Converting Text to Numerical vectors - BOW Representation

# In[55]:


X_train.head()


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer

vocab = CountVectorizer()

X_train_bow = vocab.fit_transform(X_train['clean_text_lemma'])


# In[57]:


get_ipython().run_line_magic('time', "X_train_bow = vocab.fit_transform(X_train['clean_text_lemma'])")


# In[58]:


print("Total unique words:", len(vocab.vocabulary_))

print("Type of train features:", type(X_train_bow))

print("Shape of input data:", X_train_bow.shape)


# In[59]:


print(X_train_bow.toarray())


# In[60]:


from sys import getsizeof

print(type(X_train_bow))
print(getsizeof(X_train_bow), "Bytes")


# # Term Frequency Inverse Document Frequency(TF-IDF)

# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

dtm = vectorizer.fit_transform(X_train['clean_text_lemma'])


# In[62]:


print(vectorizer.vocabulary_)


# In[63]:


print(dtm.toarray())


# In[64]:


pd.DataFrame(dtm.toarray(), columns=sorted(vectorizer.vocabulary_))


# In[65]:


X_train_TF_IDF = vectorizer.fit_transform(X_train['clean_text_lemma'])


# # Converting Text to Numerical vectors - Word2Vec Representation

# In[66]:


import gensim
import numpy as np

print(gensim.__version__)


# In[67]:


X_train['tokenised_sentences'] = X_train.clean_text_lemma.apply(lambda sent : sent.split())

X_train.head()


# In[68]:


from gensim.models import Word2Vec


# In[69]:


model = Word2Vec(list(X_train.tokenised_sentences), vector_size=300, min_count=1)


# In[70]:


print(model)


# In[71]:


# Checking the shape of vectors learned by the model

print(model.wv.__getitem__(model.wv.index_to_key).shape)


# In[72]:


def document_vector(doc, keyed_vectors):
    """Remove out-of-vocabulary words. Create document vectors by averaging word vectors."""
    vocab_tokens = [word for word in doc if word in keyed_vectors.index_to_key]
    return np.mean(keyed_vectors.__getitem__(vocab_tokens), axis=0)


# In[73]:


X_train['doc_vector'] = X_train.tokenised_sentences.progress_apply(lambda x : document_vector(x, model.wv))


# In[74]:


X_train.head()


# In[75]:


X_train_w2v = list(X_train.doc_vector)


# # Pretrained BERT for Sentence Vectors

# In[76]:


get_ipython().system(' pip install -U sentence-transformers')


# In[77]:


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')


# In[78]:


X_train['doc_vector_pretrained_bert'] = X_train.clean_text_lemma.progress_apply(model.encode)


# In[79]:


X_train.head()


# In[80]:


X_train_bert_pretrained = list(X_train.doc_vector_pretrained_bert)


# # Preprocessing The Test Data

# In[81]:


X_test.head()


# In[82]:


temp_df = X_test['review_text'].progress_apply(lambda x: preprocess(x, 'lemma'))

temp_df.head()


# In[83]:


temp_df.columns = ['clean_text_lemma', 'text_length_lemma']

temp_df.head()


# In[84]:


X_test = pd.concat([X_test, temp_df], axis=1)

X_test.head()


# BOW

# In[85]:


X_test_bow = vocab.transform(X_test['clean_text_lemma'])


# In[86]:


X_test_bow


# TF-IDF

# In[87]:


X_test_TF_IDF=vectorizer.fit_transform(X_test['clean_text_lemma'])


# In[88]:


X_test_TF_IDF


# W2V

# In[89]:


X_test['tokenised_sentences'] = X_test.clean_text_lemma.apply(lambda sent : sent.split())

X_test.head()


# In[90]:


def document_vector(doc, keyed_vectors):
    """Remove out-of-vocabulary words. Create document vectors by averaging word vectors."""
    vocab_tokens = [word for word in doc if word in keyed_vectors.index_to_key]

    if not vocab_tokens:
        # Handle empty vocab_tokens, e.g., return a zero vector
        return np.zeros(keyed_vectors.vector_size)

    return np.mean(keyed_vectors.__getitem__(vocab_tokens), axis=0)


# In[91]:


X_test['doc_vector'] = X_test.tokenised_sentences.progress_apply(lambda x : model.encode(x))


# In[92]:


X_test.head()


# In[93]:


X_test_w2v = list(X_test.doc_vector)


# In[94]:


X_test_w2v


# BERT

# In[95]:


X_test['doc_vector_pretrained_bert'] = X_test.clean_text_lemma.progress_apply(model.encode)


# In[96]:


X_test_bert_pretrained = list(X_test.doc_vector_pretrained_bert)


# Building A Model

# Pipeline for Text data

# In[97]:


data


# In[98]:


data['sentiment'].value_counts(normalize=True)


# In[99]:


X=data['review_text']
y=data['sentiment']
print(X.shape, y.shape)


# Split the Data into Train and Test

# In[100]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[101]:


import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


# In[102]:


def clean(doc): # doc is a string of text
    # This text contains a lot of <br/> tags.
    doc = doc.replace("</br>", " ")

    # Remove punctuation and numbers.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])

    # Converting to lower case
    doc = doc.lower()

    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]

    # Join and return
    return " ".join(filtered_tokens)


# In[103]:


from sklearn.feature_extraction.text import CountVectorizer

# instantiate a vectorizer
vect = CountVectorizer(preprocessor=clean)

# use it to extract features from training data
get_ipython().run_line_magic('time', 'X_train_dtm = vect.fit_transform(X_train)')

print(X_train_dtm.shape)


# In[104]:


X_test_dtm = vect.transform(X_test)

print(X_test_dtm.shape)


# In[105]:


X_train_clean = X_train.apply(lambda doc: clean(doc))


# In[106]:


X_test_clean = X_test.apply(lambda doc: clean(doc))


# In[107]:


import joblib
from joblib import Memory
import seaborn as sns
import os
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np


# In[108]:


import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# In[109]:


cachedir = '.cache'
memory = Memory(location=cachedir, verbose=0)

pipelines = {
    'naive_bayes': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', MultinomialNB())
    ], memory=memory),
    'decision_tree': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', DecisionTreeClassifier())
    ], memory=memory),
    'logistic_regression': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', LogisticRegression())
    ], memory=memory)
}

# Define parameter grid for each algorithm
param_grids = {
    'naive_bayes': [
        {
            'vectorization': [CountVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__alpha' : [1, 10]
        }
    ],
    'decision_tree': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'logistic_regression': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['elasticnet'],
            'classifier__l1_ratio': [0.4, 0.5, 0.6],
            'classifier__solver': ['saga'],
            'classifier__class_weight': ['balanced']
        }
    ]
}

# Perform GridSearchCV for each algorithm
best_models = {}

for algo in pipelines.keys():
    print("*"*10, algo, "*"*10)
    grid_search = GridSearchCV(estimator=pipelines[algo],
                               param_grid=param_grids[algo],
                               cv=5,
                               scoring='f1',
                               return_train_score=True,
                               verbose=1
                               )

    grid_search.fit(X_train_clean, y_train)

    best_models[algo] = grid_search.best_estimator_

    # Calculate F1 score manually
    from sklearn.metrics import f1_score
    y_pred = grid_search.predict(X_test_clean)
    weighted_f1_score = f1_score(y_test, y_pred, average='weighted')

    print('Score on Test Data: ', weighted_f1_score)


# In[111]:


y_train_pred = grid_search.predict(X_train)


# In[112]:


cm = confusion_matrix(y_train, y_train_pred)
actual = np.sum(cm, axis=1).reshape(-1, 1)
cmn = np.round(cm / actual, 2)

# Plot heatmap for train data
plt.figure(figsize=(8, 6))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=best_models[algo].classes_, yticklabels=best_models[algo].classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap {}'.format(algo))
plt.show()


# In[113]:


y_test_pred = grid_search.predict(X_test)


# In[114]:


cm = confusion_matrix(y_test, y_test_pred)
actual = np.sum(cm, axis=1).reshape(-1, 1)
cmn = np.round(cm / actual, 2)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=best_models[algo].classes_, yticklabels=best_models[algo].classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap - {}'.format(algo))
plt.show()


# In[115]:


for name, model in best_models.items():
    print(f"{name}")
    print(f"{model}")
    print()


# In[125]:


best_model=grid_search.best_estimator_


# In[126]:


new_data=["worst"]


# In[127]:


new_data_clean = [clean(doc) for doc in new_data]


# In[128]:


prediction = model.predict(new_data_clean)

print("Prediction:", prediction)


# In[124]:


import os
from IPython.display import FileLink

# Define the file path of your pickle file
file_path = 'P:\\INNMOATICS\\FLASK\\ML_FLASK_DEPLOYMENT.pkl'

# Check if the file exists
if os.path.exists(file_path):
    # Generate a download link for the file
    display(FileLink(file_path))
else:
    print("The file does not exist.")


# In[ ]:




