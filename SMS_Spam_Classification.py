#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'messages'])


# In[3]:


dataset


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


dataset['label'] = dataset['label'].map({'ham':0, 'spam': 1})


# In[7]:


dataset


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


plt.figure(figsize=(8,8))
g = sns.countplot(x='label', data=dataset)
p = plt.title('Countplot for Spam vs Ham as imbalanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')


# In[10]:


only_spam = dataset[dataset['label'] == 1]


# In[11]:


only_spam


# In[12]:


print('No. of Spam SMS:', len(only_spam))
print('No. of Ham SMS:', len(dataset) - len(only_spam))


# In[13]:


count = int((dataset.shape[0] - only_spam.shape[0])/ only_spam.shape[0])


# In[14]:


count


# In[15]:


for i in range (0,count-1):
    dataset = pd.concat([dataset, only_spam])


# In[16]:


dataset.shape


# In[17]:


plt.figure(figsize=(8,8))
g = sns.countplot(x='label', data=dataset)
p = plt.title('Countplot for Spam vs Ham as balanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')


# In[18]:


dataset['word_count'] = dataset['messages'].apply(lambda x: len(x.split()))


# In[19]:


dataset


# In[20]:


plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
g = sns.histplot(dataset[dataset["label"] == 0].word_count, kde=True)
p = plt.title("Distribution of word_count for Ham SMS")

plt.subplot(1,2,2)
g = sns.histplot(dataset[dataset["label"] == 1].word_count, kde=True)
p = plt.title("Distribution of word_count for Ham SMS")

plt.tight_layout()
plt.show()


# In[21]:


def currency(data):
    currency_symbols = ['$', '€', '₹']
    for i in currency_symbols:
        if i in data:
            return 1
    return 0


# In[22]:


dataset["contains_currency_symbols"] = dataset['messages'].apply(currency)


# In[23]:


dataset


# In[24]:


plt.figure(figsize=(8,8))
g = sns.countplot(x='contains_currency_symbols', data=dataset, hue = 'label')
p = plt.title('Countplot for containing currency symbol')
p = plt.xlabel("Does SMS contains any currency symbol?")
p = plt.ylabel('count')
p = plt.legend(labels=["Ham", "Spam"], loc= 9)


# In[25]:


def number(data):
       for i in data:
           if ord(i)>= 48 and ord(i) <= 57:
               return 1
       return 0


# In[26]:


dataset["contains_number"] = dataset['messages'].apply(number)


# In[27]:


dataset


# In[28]:


plt.figure(figsize=(8,8))
g = sns.countplot(x='contains_number', data=dataset, hue = 'label')
p = plt.title('Countplot for containing currency symbol')
p = plt.xlabel("Does SMS contains any number?")
p = plt.ylabel('count')
p = plt.legend(labels=["Ham", "Spam"], loc= 9)


# In[29]:


import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[30]:


import nltk
nltk.download('omw-1.4')

corpus = []
wnl = WordNetLemmatizer()

for sms in list(dataset.messages):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms) 
    message = message.lower()
    words = message.split()
    filtered_words = [word for word in message if word not in set(stopwords.words('english'))] 
    lemm_words = [wnl.lemmatize(word) for word in message]
    message = ''.join(message)
    
    corpus.append(message)


# In[31]:


corpus


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()


# In[33]:


X = pd.DataFrame(vectors, columns = feature_names)
y = dataset['label']


# In[34]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state= 42)


# In[36]:


X_test


# In[37]:


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
cv = cross_val_score(mnb, X, y, scoring='f1', cv=10)
print(round(cv.mean(),3))
print(round(cv.std(),3))


# In[38]:


mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)


# In[39]:


classification_report(y_test, y_pred)


# In[40]:


print(classification_report(y_test, y_pred))


# In[41]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[42]:


plt.figure(figsize=(8,8))
axis_labels = ['Ham', 'Spam']
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels)
p = plt.title("confusion matrix of multinomial naive bayes model")
p = plt.xlabel('actual values')
p = plt.ylabel('predicted values')


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv1 = cross_val_score(dt, X, y, scoring='f1', cv=10)
print(round(cv1.mean(),3))
print(round(cv1.std(),3))


# In[44]:


dt.fit(X_train, y_train)
y_pred1 = dt.predict(X_test)


# In[45]:


print(classification_report(y_test, y_pred))


# In[46]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[49]:


def predict_spam(sms):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms) 
    message = message.lower()
    words = message.split()
    filtered_words = [word for word in message if word not in set(stopwords.words('english'))] 
    lemm_words = [wnl.lemmatize(word) for word in message]
    message = ''.join(message)
    temp = tfidf.transform([message]).toarray()
    return mnb.predict(temp)


# In[50]:


sample_message = 'You have won a lottery of 3000 $'

if predict_spam(sample_message):
    print("Gotcha! This is a Spam Message ")
    
else:
    print("The message is Ham")


# In[ ]:




