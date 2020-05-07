
# READ BASICS FROM NAIVE BAYES ALSO READ ABOUT LAPLACE SMOOTING
import numpy as np
import pandas as pd
import sklearn

docs = pd.read_csv("D:/dataforpython/NB_train.xlsx.csv")   # text in column 1 , classifier in col 2

# convert label to a numeric var
docs["CLASS"] = docs.CLASS.map({"cinema" : 0,"education" : 1})

# taking out x and y
x = docs.iloc[:,0].values
y = docs.iloc[:,1].astype("int").values
print(x)

# create an object of count vectorizer
# here we are going to create an object vec of class count vectorization(), this has a method called as fit() which converts a corplus of documents into a vector of unique
# words as shown below
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()

vec.fit(x)
vec.vocabulary_      # here we have counted ach word as a vector

# STOP WORDS : eg : and,if ,of this words really dont make any diff in classfying a problem
# removing stop words
vec = CountVectorizer(stop_words="english")
vec.fit(x)
vec.vocabulary_

# printing feature names
print(vec.get_feature_names())
print(len(vec.get_feature_names()))

# so our final dictonary is made of 12 words (after disacrding stop words) now to do classification we need to represent all the documents w.r.t these words in
# form of features
# every document will be converted into feature vector representing presence of words in that document

# another way of representing feature
x_transform = vec.transform(x)
x_transform

print(x_transform)

x = x_transform.toarray()
x

pd.DataFrame(x,columns=vec.get_feature_names())
x

# testing
test_docs = pd.read_csv("D:/dataforpython/NB_test.csv")
test_docs

# convert label to numerical
test_docs["CLASS"] = test_docs["CLASS"].map({"cinema":0,"education":1})
x_test = test_docs.iloc[:,0].values
y_test = test_docs.iloc[:,1].astype("int").values

x_test_transform = vec.transform(x_test)
x_test_transform

x_test = x_test_transform.toarray()

# building multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB

# instances NB class
mnb = MultinomialNB()

# fitting model
mnb.fit(x,y)

# predicting probabilities of test data
mnb.predict_proba(x_test)
probability = mnb.predict_proba(x_test)

print("probability of test doc belonging to class cinema is" + str(probability[:,0]))
print("probability of test doc belonging to class education is" + str(probability[:,1]))

# bernoulli naive bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x,y)
bnb.predict_proba(x_test)
