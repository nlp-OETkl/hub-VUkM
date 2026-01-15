import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
#这是个逻辑回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 读取数据（按空格分割，但注意：文本中可能有多个空格，所以只分割最后一部分作为标签）
def load_data(file_path):
    """
    用于确认数据集的真实分隔符
        with open(file_path, 'rb') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"原始字节 {i + 1}: {line}")
            print(f"解码后 {i + 1}: {line.decode('utf-8', errors='replace')}")
    
    """
    # 明确指定 sep='\t' 表示用制表符分隔
    df = pd.read_csv(
        file_path,
        sep='\t',  #  关键：制表符分隔
        header=None,  # 文件没有列名
        names=['text', 'label'],  # 自定义列名
        encoding='utf-8',  # 中文通常 UTF-8
        skip_blank_lines=True  # 跳过空行
    )

    # 去除可能的前后空格（防止标签有空格）
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()

    # 过滤掉空文本或空标签的行
    df = df[(df['text'] != '') & (df['label'] != '')]

    return df['text'].tolist(), df['label'].tolist()

# 加载数据
texts, labels = load_data('dataset.csv')

# 检查数据是否加载成功
print(f"共加载 {len(texts)} 条数据")
print("前3条示例:")
for i in range(min(3, len(texts))):
    print(f"文本: '{texts[i]}' → 标签: {labels[i]}")

# 定义中文分词函数
def chinese_tokenizer(text):
    return " ".join(jieba.cut(text))

# 对文本进行分词（用于 TfidfVectorizer）
tokenized_texts = [chinese_tokenizer(text) for text in texts]

# Pipeline 实现工作流的类，用于将多个数据处理步骤（transformers）和一个最终的估计器（通常是模型）串联成一个整体
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=500))
])

# 训练模型
pipeline.fit(tokenized_texts, labels)

def FenXi(text:list):
    test_sentence = [" ".join(jieba.lcut(x)) for x in text]
    return pipeline.predict(test_sentence)

# 示例预测
if __name__ == "__main__":
    sentence = input("请输入需要预测的语句(回车代表下一句，若需要暂停，请输入end)：")
    sentences = []
    while sentence != "end":
        sentences.append(sentence)
        sentence = input("请输入下一句（输入end）：")
    predictions = FenXi(sentences)
    for sent, pred in zip(sentences, predictions):
        print(f"输入: {sent} → 预测类别: {pred}")