import numpy as np
import pandas as pd
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, accuracy_score
from sklearn.decomposition import TruncatedSVD

# Data downloading script

########
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if ('dataset.csv' not in os.listdir('../Data')):
    print('Dataset loading.')
    url = "https://www.dropbox.com/s/0sj7tz08sgcbxmh/large_movie_review_dataset.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/dataset.csv', 'wb').write(r.content)
    print('Loaded.')
# The dataset is saved to `Data` directory
########

# write your code here
df = pd.read_csv('../Data/dataset.csv')
count_before = df.shape[0]
df = df[~((df.rating >= 5.0) & (df.rating <= 7.0))]
df.loc[df.rating > 7.0, 'label'] = 1
df.loc[df.rating < 5.0, 'label'] = 0
df.drop(columns=['rating'], inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(df.review, df.label.to_numpy(), random_state=23)
vectorizer = TfidfVectorizer(sublinear_tf=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

svd = TruncatedSVD(n_components=100)
X_train_svd = svd.fit_transform(X_train_vectorized)
X_test_svd = svd.transform(X_test_vectorized)

logistic = LogisticRegression(solver='liblinear', penalty='l1', C=0.15)
logistic.fit(X_train_svd, Y_train)

yhat = logistic.predict_proba(X_test_svd)

accuracy = accuracy_score(Y_test, pd.DataFrame(yhat.T[1]).round().to_numpy())
auc_score = roc_auc_score(Y_test, yhat.T[1])

# feature_count = 0
# for i in logistic.coef_[0]:
#     if np.absolute(i) > 0.0001:
#        feature_count += 1


print(f'{accuracy:.2f}')
print(f'{auc_score:.2f}')
# print(feature_count)

