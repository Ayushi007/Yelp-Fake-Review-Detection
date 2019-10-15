import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot
import scikitplot.plotters as skplt
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import zero_one_loss

merge_df = pd.read_csv("path/cleandata.csv") #delimiter = '\t',header=None,names=["user_id","prod_id","date","review"])

merge_df.dropna(inplace = True)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(merge_df["review_clean"].apply(lambda x: x.split(" ")))]
documents[0]

# # train a Doc2Vec model with our text data
# model = Doc2Vec(documents[0], vector_size=5, window=2, min_count=1, workers=4)

# # transform each document into a vector data
# doc2vec_df = merge_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
# doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
# merge_df2 = pd.concat([merge_df, doc2vec_df], axis=1)

# merge_df.head()

document_list = []
for item in documents:
    document_list.append(item[0])

merge_df["doc_words"]=document_list

#Fastext model defined
model = FastText(size=5, window=3, min_count=1)
model.build_vocab(document_list)
model.train(document_list, total_examples=model.corpus_count, epochs=model.epochs, size=5, window=3)

sum_v = np.zeros(len(model.wv['drink']))
i = 0
for item in sum_v:
    col_name = "v"+str(i)
    merge_df[col_name] = np.zeros(merge_df["date"].shape)
    i = i+1

sum_list = []
i = 0
for doc in document_list:
    for each in doc:
        sum_v = sum_v + model.wv[each]
    sum_list.append(sum_v)
    i = i+1
    print("iteration - ",i)


df_sum_v = pd.DataFrame(sum_list,columns = ["v0","v1","v2","v3","v4"])
i = 0
for item in sum_v:
    col_name = "v"+str(i)
    merge_df[col_name] = df_sum_v[col_name]
    i = i+1

merge_df["count"]  = [len(c) for c in merge_df['doc_words']]

for i in range(5):
    col_name = "v"+str(i)
    col_name_avg = col_name + "_avg"
    merge_df[col_name_avg] = merge_df[col_name]/merge_df["count"]

# merge_df = merge_df.drop(["v5","v6","v7"], axis = 1)

# merge_df.to_csv("FastText5(size 5).csv",index=False)

label = "label"
ignore_cols = [label, "review", "review_clean", "doc_words", "date", "pos_tags", "pos", "dict_pos"]
features = [c for c in merge_df.columns if c not in ignore_cols]

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(merge_df[features], merge_df[label], test_size = 0.20, random_state = 20)

# X_train.head()
# X_train = X_train.drop("doc_words",axis = 1)
# X_train = X_train.drop("date",axis = 1)
# X_train = X_train.drop("pos_tags",axis = 1)
# X_train = X_train.drop("pos",axis = 1)
# X_train = X_train.drop("dict_pos",axis = 1)

X_train = np.nan_to_num(X_train)

# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 20, random_state = 42)

rf.fit(X_train, y_train)

# feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
# feature_importances_df.head(20)

# X_test.loc[0]
# X_test = X_test.drop("doc_words",axis = 1)
# X_test = X_test.drop("date",axis = 1)
# X_test = X_test.drop("pos_tags",axis = 1)
# X_test = X_test.drop("pos",axis = 1)
# X_test = X_test.drop("dict_pos",axis = 1)
# X_test.loc[0]

X_test = np.nan_to_num(X_test)

y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('FastText.png')
plt.show()

y_pred_class = rf.predict(X_test)
zero_one_loss(y_test, y_pred_class)

list_score = rf.score(X_test, y_test)

skplt.plot_confusion_matrix(y_test, y_pred_class, normalize=True)
plt.savefig('confusion_word2vec.png')
plt.show()
