import jieba  # 中文分词的用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree  # 决策树模块


def text_classify_ml(txt: str) -> str:
    data = pd.read_csv("dataset.csv", sep="\t", names=["text", "label"], nrows=1000)

    data_set = data["text"].apply(lambda x: " ".join(jieba.lcut(x)))

    vector = CountVectorizer()
    vector.fit(data_set)
    input_feature = vector.transform(data_set)

    model = KNeighborsClassifier()
    model.fit(input_feature, data["label"].values)

    input_set = " ".join(jieba.lcut(txt))
    feature = vector.transform([input_set])
    return model.predict(feature)


def text_classify_ml_tree(txt: str) -> str:
    data = pd.read_csv("dataset.csv", sep="\t", names=["text", "label"], nrows=1000)

    data_set = data["text"].apply(lambda x: " ".join(jieba.lcut(x)))

    vector = CountVectorizer()
    vector.fit(data_set)
    input_feature = vector.transform(data_set)

    model = tree.DecisionTreeClassifier()
    model.fit(input_feature, data["label"].values)

    input_set = " ".join(jieba.lcut(txt))
    feature = vector.transform([input_set])
    return model.predict(feature)

if __name__ == '__main__':
    # 读取数据
    # pandas 用来进行表格的加载和分析
    # numpy 从矩阵的角度加载和计算
    # data = pd.read_csv("dataset.csv", sep="\t", names=["text", "label"], nrows=1000)
    # print(data.head(10))
    # print("数据集的样本维度：",data.shape)
    # print(data["label"].value_counts()) # label 列的频次分布

    # jieba.add_word("机器学习")
    # print(jieba.lcut("我今天开始学习机器学习。"))
    test_query = "我想导航到天安门广场"
    print(test_query, " KNN :", text_classify_ml(test_query))
    print(test_query, " TREE :", text_classify_ml_tree(test_query))
