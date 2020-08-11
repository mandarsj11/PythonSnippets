text1 = "Ethics are built right into the ideals and objectives of the United Nations "
len(text1) # The length of text1
list(text1) #gives list of all characters
[c for c in text1] #gives list of all characters
text2 = text1.split(' ') # Return a list of the words in text2, separating by ' ', duplicate words are considered seperate.
[w for w in text2 if len(w) > 3] # Words that are greater than 3 letters long in text2
[w for w in text2 if w.istitle()] # Capitalized words in text2
[w for w in text2 if w.endswith('s')] # Words in text2 that end in 's'

text3 = 'To be or not to be'
text4 = text3.split(' ')
set(text4) # Finding unique words/letters in sentence/word/list
len(set(text4))

set([w.lower() for w in text4]) # .lower converts the string to lowercase.
len(set([w.lower() for w in text4])) 

text5 = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'
text6 = text5.split(' ')

text6

#comparison operations
[w for w in text6 if w.startswith('#')]
[w for w in text6 if w.endswith('s')]
[w for w in text6 if ('t') in w]
[w for w in text6 if w.isupper()]
[w for w in text6 if w.islower()]
[w for w in text6 if w.istitle()]
[w for w in text6 if w.isalpha()]
[w for w in text6 if w.isdigit()]
[w for w in text6 if w.isalnum()] #alphabets OR numericals

#modification operation
[w.lower() for w in text6]
[w.upper() for w in text6]
[w.titlecase() for w in text6] # does not work - 'str' object has no attribute 'titlecase'
[w.split('a') for w in text6]

['z'.join(w) for w in text6]
[w.strip('"') for w in text6]
w.rstrip()
[w.find('a') for w in text6]
[w.rfind('a') for w in text6]
[w.replace('a','XX') for w in text6]

#cleaning text
f = open('Embezzled.txt','r')
f.readline() #reads till next new line character \n
f.readline().rstrip() #strip blank characters at the start and end and \n
f.seek(0) #initialise the reading pointer to nth place
text12 = f.read()
text13 = text12.splitlines #splits on '\n'
text14 = f.read(100) #read n characters in the file
f.write(message)
f.close()
f.closed

##3333333333333333333333333333333333333333333333333333333333
#Processing free text
text7 = '@UN @UN_Women "Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'
text8 = text7.split(' ')

# We can use regular expressions to help us with more complex parsing. 
# 
# For example `'@[A-Za-z0-9_]+'` will return all words that: 
# * start with `'@'` and are followed by at least one: 
# * capital letter (`'A-Z'`)
# * lowercase letter (`'a-z'`) 
# * number (`'0-9'`)
# * or underscore (`'_'`)

import re # import re - a module that provides support for regular expressions

[w for w in text8 if re.findall('[^gh]', w)] #find specific matching characters
[[w, re.findall('[ti]{2}', w)] for w in text8] 

[w for w in text8 if re.search('@c+', w)] #search match location
[[w, re.search('[oti]{3}', w)] for w in text8] - #can search 'Nations', but not 'objectives'

#Character Match - What?
. - wildcard, match a single character
^ - start of a string
$ - end of a string
[] - match one of the set of characters within []
[a-z] - match range of characters
[^abc] = does not match abc
a|b - either a or b
() - scoping of operators
\ - escape character (\t, \n, \b)
\b - word boundary
\d - Any digit, [0-9]
\D - Any non digit [^0-9]
\s - Any whitespace character [\t\n\r\f\v]
\S - Any non whitespace character [^\t\n\r\f\v]
\w - Alphanumeric character [A-Za-z0-9_]
\W - non Alphanumeric character [^A-Za-z0-9_]

#Match how many times - frequency 
* - no match or matches multiple times
+ - match one or more times
? - does not match or one match
{n} - match exactly n times n>=0
{n,} - match at least n times
{,n} - match at most n times
{m,n} - at least m and at most n times
?: - 

#Dat examples:
'\d{2}[/-]\d{2}[/-]\d{4}'
'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
'\d{2} (Jan|feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}' #pulls out only the match in ()
'\d{2} (?:Jan|feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}' #pulls entire string
'(?:\d{2} )?(?:Jan|feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{2}, )?\d{4}'
'(?:\d{1,2} )?(?:Jan|feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{1,2}, )?\d{2,4}'


import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
df

# find the number of characters for each string in df['text']
df['text'].str.len()

# find the number of tokens for each string in df['text']
df['text'].str.split().str.len()

# find which entries contain the word 'appointment'
df['text'].str.contains('appointment')

# find how many times a digit occurs in each string
df['text'].str.count(r'\d')

# find all occurances of the digits
df['text'].str.findall(r'\d')

# group and find the hours and minutes
df['text'].str.findall(r'(\d?\d):(\d\d)')

# replace weekdays with '???'
df['text'].str.replace(r'\w+day\b', '???')

# replace weekdays with 3 letter abbrevations
df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])

# create new columns from first match of extracted groups
df['text'].str.extract(r'(\d?\d):(\d\d)')

# extract the entire time, the hours, the minutes, and the period
df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')

# extract the entire time, the hours, the minutes, and the period with group names
df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')


#Module 2.1 - Basic NLP Tasks with NLTK - Natural Language Toolkit
import nltk
nltk.download()  #location to verify - C:\Users\dell\AppData\Roaming\nltk_data
from nltk.probability import FreqDist
from nltk.book import *
#1 - Counting vocabulary of words
len(sent7)
len(text1)
len(set(text7)) #unique words
list(set(text7))[:10]
dist = FreqDist(text7) #Frequency of words
len(dist)
#anguments for frequency distribution
vocab1 = dist.keys() #actual words
dist.max()
dist.freq('whale') #count of that sample divided by the total number of sample outcomes that have been recorded by this FreqDist.
dist.most_common(20) #cost common x words
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
list(vocab1)[:10]

dist[s''four'] #frequency of perticulat word
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
freqwords

#Word frequency distribution for a text in a dataframe
freq_dist = df.text.str.split(expand=True).stack().value_counts()

[index for index, value in freq_dist.items() if len(index) >5 and int(value)>2] 
    

for index, value in freq_dist.items():
    print(len(index),value)

#2 - Normalization (diff forms of same word) and stemming (Identify root form of the word) - 
input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')

porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1] # Identify root form of the word


#3 - Lemmatization - Identify root form of word that is meaningful word
udhr = nltk.corpus.udhr.words('English-Latin1')
udhr[:20]

[porter.stem(t) for t in udhr[:20]] # using porter.stemmer

WNlemma = nltk.WordNetLemmatizer() # better stemmatizer
[WNlemma.lemmatize(t) for t in udhr[:20]]

#4 - Tokenization - words & sentences
text11 = "Children shouldn't drink a sugary drinking before bed. good better best"
text11.split(' ')

nltk.word_tokenize(text11) #finding words within sentence

text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12) #finding sentences within a paragraph
len(sentences)
sentences

#Module 2.2 - Advanced NLP tasks with NLTK
#1 - Part-of_speech (POS) tagging

nouns - NN - sigular, plural, proper
verbs - VB - gerunds, past tense verbs
adjectives - JJ - 0
conjunctions - CC
cardinals - CD - if you have a number, then you are to kind of assign that word class
determiner - DT -
prepositions - IN -
modal - MD - 
possessives - POS -
pronouns - PRP - 
adverbs - RB - 
symbols - SYM - 

nltk.help.upenn_tagset('MD') #get explaination about POS

text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13) #POS tagging for every word in a sentence

#2 - Ambiguity of POS tagging
text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text14)
tag_fd = nltk.FreqDist(tag for (word, tag) in postag) #frequency of POS tagging
tag_fd.most_common(5)

#More info - https://www.nltk.org/book/ch05.html

#3 Parsing sentence structure to plot Context Free Grammer tree (CFG)
text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)

#4 - Ambiguity in parsing sentences
text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('mygrammar.cfg')
grammar1

parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)
    
from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)

#5 - POS tagging and parsing ambiguity
#meaningful sentence without proper structure
text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)

#well formed sentence but meaningless
text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)

# Types of Naive Bayes Classifiers
from sklearn.naive_bayes import MultinomialNB #word occurance or frequency is important
from sklearn.naive_bayes import BernoulliNB #word presence of absence is important - presence of 'the' or 'Parlament' is treated equally
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(df['Reviews'], 
                                                    df['Positively Rated'], test_size = 0.333,
                                                    random_state=0)

predicted_label = model_selection.cross_val_predict(model, df['Reviews'], df['Positively Rated']),
                                                    cv=5)

#Module 3.1 Classification of Text

#Workflow/Pipeline - Case Study: Sentiment Analysis

#Step 1 - Data Prep
import pandas as pd
import numpy as np

# Read in the data
df = pd.read_csv('Amazon_Unlocked_Mobile.csv')

# Sample the data to speed up computation
# Comment out this line to match with lecture
df = df.sample(frac=0.1, random_state=10)

df.head()

# Drop missing values
df.dropna(inplace=True)

# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]

# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)

# Most ratings are positive
df['Positively Rated'].mean()

from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positively Rated'], 
                                                    random_state=0)

print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)

#Step 2 - CountVectorizer - converting to numerical representation
#Bag of words approach - count how often each word occurs
#involves tokenization of training data and build vocabulary
#convert to document term Metrix - 
from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)

vect.get_feature_names()[::2000]
len(vect.get_feature_names())

# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)

X_train_vectorized

from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

#Step 3 - Tfidf: Term Frequency Inverse Document Frequency
#weight terms/features based on how important they are to the document
#document frequency - no fo rows in a dataframe

#Low Tfidy - Words, commonly occur across all documents OR
#            Occur very rarely
                
#High Tfidf - Words, occur frequently in specific documents but, 
#             rarely used across all documents

#min_df - min no of documents the word should occure to become part of the vocabulary

#Term Frequency: is a scoring of the frequency of the word in the current document.
#TF = (Number of times term t appears in a document)/(Number of terms in the document)

#Inverse Document Frequency: is a scoring of how rare the word is across documents.
#IDF = 1+log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.

#Tf-IDF weight = Term Frequency * Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5

vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


#Step 4 - n-grams: add sequence of word features
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

len(vect.get_feature_names())

X_train_vectorized = vect.transform(X_train)

#Week 4 -
#Semantic similarity
#Textual entailment - whether a sentence derives os meaning from a paragraph or not
#paraphrasing - re-writing sentence which has same meaning as original

#Method 1 - path similarity - finding shortest path - 1/(separation + 1)

import nltk
from nltk.corpus import wordnet as wn

deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')
deer.path_similarity(elk)
deer.path_similarity(horse)
elk.path_similarity(deer)

#Method 2 - Lowest common subsumer(LCS) - closest ancsestor to both
#Method 3 - Lin similarity - 

nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
deer.lin_similarity(elk,brown_ic)
deer.lin_similarity(horse,brown_ic)


#Method 4 - Colocation and distribution similarity
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(text)
finder.nbest(bigram_measures.pmi,10)

#frequency filter
finder.apply_freq_filter(10)

#Topic Modelling




