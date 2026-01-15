import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=1000)


# 中文分词并拼接成空格分隔的字符串（适配 CountVectorizer）
input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(str(x))))  # 防止非字符串
labels = dataset[1].values

# 文本向量化
vectorizer = CountVectorizer()

# 划分训练集和测试集，训练：测试8:2
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    input_sentences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # 确保各类别比例一致（适用于分类）
)

# 只在训练集上拟合 vectorizer（防止数据泄露！）
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# 训练 KNN 模型
# 这个参数相当于调整模型精度，越小精度越高，但是会导致模型过于敏感，越大越平滑，但会导致精度缺失
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 在测试集上预测
y_pred = model.predict(X_test)

# 输出评估结果
print("\n=== 模型评估结果 ===")
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 预测函数（使用已训练好的 vectorizer 和 model）
def FenXi(text:list):
    test_sentence = [" ".join(jieba.lcut(x)) for x in text]
    test_feature = vectorizer.transform(test_sentence)
    return model.predict(test_feature)

# 示例预测
if __name__ == "__main__":
    sentence = input("请输入需要预测的语句(回车代表下一句，若需要暂停，请输入end)：")
    sentences = []
    sentences.append(sentence)
    while sentence != "end":
        sentence = input("请输入下一句（输入end）：")
        sentences.append(sentence)
    print(FenXi(sentences))