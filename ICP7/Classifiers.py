from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

#problem1
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
classifier = SVC(kernel='linear', random_state=0)

clf.fit(X_train_tfidf, twenty_train.target)
classifier.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted =clf.predict(X_test_tfidf)
predicted1 = classifier.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
score1 = metrics.accuracy_score(twenty_test.target, predicted1)
print("accuracy score with multinomialNB",score)
print("accuracy score after applyingSVM",score1)


#problem2
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer(ngram_range=(2,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted =clf.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy score by applying bigram",score)

#problem3
#stop_words = set(stopwords.words('english'))
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer(stopwords.words('english'))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted =clf.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy score by applying stopword",score)