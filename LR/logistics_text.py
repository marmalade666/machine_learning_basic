print("开始....")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

df_train = pd.read_csv("./train_set.csv")
df_test = pd.read_csv("./test_set.csv")
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000) #词频作为特征，也可以使用TfidfVectorizer
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class'] - 1

lg = LogisticRegression(C=4, dual=True) #C是惩罚函数的倒数，值越小，正则化越大；penalty指定正则化策略，solver求解最优化问题的算法等
lg.fit(x_train, y_train)

y_test = lg.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']] #保留有用的
df_result.to_csv('./result.csv', index=False)

print('完成......')