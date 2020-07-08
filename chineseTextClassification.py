from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import mpl
import numpy as np
import jieba

"""
特征工程：
（1）jieba分词
（2）词频统计
时政 63085
体育 131604
娱乐 92631
游戏 24372
星座 3577
"""

jieba.load_userdict(r"D:\PycharmProjects\ChineseText_classification\data\wordDict.txt")  # 导入词库


def train_dataset():
    """ 加载训练集 """
    train_data = pd.read_csv(r"D:\PycharmProjects\THUCNews_after_data\train_data.txt", encoding='utf8', sep='\t',
                             names=['label', 'content'])
    return train_data


def test_dataset():
    """ 加载测试集 """
    test_data = pd.read_csv(r"D:\PycharmProjects\THUCNews_after_data\test_data.txt", encoding='utf8', sep='\t',
                            names=['label', 'content'])
    return test_data


# def read_category(y_train):
#     """
#     文本标签转换为数字标签
#     """
#     category = ['体育', '娱乐', '时政', '星座', '游戏']
#     category_index = dict(zip(category, range(len(category))))
#     label_index = []
#     for i in range(len(y_train)):
#         label_index.append(category_index[y_train[i]])
#     return label_index


def chinese_word_cut(mytext):
    """
    jieba分词
    增加专业词汇、维护自定义词库
    """
    # jieba.add_word("word")

    print("=====分词进度=====")

    return " ".join(jieba.cut(mytext))


def get_stopwords(stop_words_file):
    """ 加载停用词表 """
    with open(stop_words_file, encoding='utf8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    after_stopwords_list = [i for i in stopwords_list]

    return after_stopwords_list


def tfidf_feature():
    """ TF-IDF提取特征 """
    train_data = train_dataset()
    test_data = test_dataset()
    stop_words = r"D:\PycharmProjects\ChineseText_classification\data\哈工大停用词表.txt"
    stopwords = get_stopwords(stop_words)
    X_train = train_data.content.apply(chinese_word_cut)
    print("分词后的训练集：\n")
    print(X_train)
    y_train = train_data['label']
    print("训练集标签：\n", list(y_train))
    X_test = test_data.content.apply(chinese_word_cut)
    print("分词后的测试集：\n")
    print(X_test)
    # 构建模型
    tfidf_vector = TfidfVectorizer(stop_words=stopwords, max_features=5000, lowercase=False, sublinear_tf=True,
                                   max_df=0.8)
    # 训练模型
    X_train_tfidf = tfidf_vector.fit_transform(X_train)  # X_train用fit_transform生成词汇表
    X_test_tfidf = tfidf_vector.transform(X_test)  # X_test要与X_train词汇表相同，因此在X_train进行fit_transform基础上进行transform操作

    print("训练集生成的词汇表维度：", X_train_tfidf.shape)
    print(X_train_tfidf.shape)
    print("测试集生成的词汇表维度：", X_test_tfidf.shape)
    print(X_test_tfidf.shape)

    return X_train, y_train, X_test, X_train_tfidf, X_test_tfidf


def nb_model():
    """ 朴素贝叶斯模型 """
    X_train, y_train, X_test, X_train_tfidf, X_test_tfidf = tfidf_feature()
    # model = MultinomialNB(alpha=0.2)  # 模型参数，可进行调优
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)  # 模型训练
    pre = model.predict(X_test_tfidf)  # 模型预测
    print("测试集预测结果：\n")
    print(pre)

    return model, X_test_tfidf, pre


def score_model():
    """ 模型评估 """
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    test_data = test_dataset()
    y_test = test_data['label']
    model, X_test_tfidf, pre = nb_model()
    print("模型评估结果：", model.score(X_test_tfidf, y_test))
    print("分类预测报告：\n", classification_report(y_test, pre))
    print("模型评估之混淆矩阵：\n", confusion_matrix(y_test, pre))
    crosstab = pd.crosstab(y_test, pre, rownames=['labels'], colnames=['predict'])
    matrix = pd.DataFrame(crosstab)
    sns.heatmap(matrix, annot=True, cmap="RdPu", fmt='d', linewidths=0.2, linecolor='pink')
    plt.show()


def crossrtab_matrix(y_test, y_pre):
    """
    交叉表、交叉矩阵
    查看预测数据与原数据对比
    """
    # y_test = np.argmax(y_test, axis=1).reshape(-1)
    # print(y_test)
    # print(type(y_test))
    # print('================')
    # print(type(y_pre))
    crosstab = pd.crosstab(y_test, y_pre, rownames=['labels'], colnames=['predict'])
    matrix = pd.DataFrame(crosstab)
    sns.heatmap(matrix, annot=True, cmap="RdPu", linewidths=0.2, linecolor='pink')
    plt.show()


if __name__ == '__main__':
    score_model()
