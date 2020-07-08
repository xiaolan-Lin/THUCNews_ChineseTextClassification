from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import jieba

# jieba.add_word("word")
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


def chinese_word_cut(mytext):
    """
    jieba分词
    增加专业词汇、维护自定义词库
    """
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
    y_train = train_data['label']
    X_test = test_data.content.apply(chinese_word_cut)
    tfidf_vector = TfidfVectorizer(stop_words=stopwords, max_features=5000, lowercase=False, sublinear_tf=True,
                                   max_df=0.8)
    tfidf_vector.fit(X_train)
    X_train_tfidf = tfidf_vector.transform(X_train)
    X_test_tfidf = tfidf_vector.transform(X_test)

    print(X_train_tfidf.shape)
    print(X_test_tfidf.shape)

    return X_train, y_train, X_test, X_train_tfidf, X_test_tfidf


def nb_model():
    """ 朴素贝叶斯模型 """
    X_train, y_train, X_test, X_train_tfidf, X_test_tfidf = tfidf_feature()
    model = MultinomialNB(alpha=0.2)  # 模型参数，可进行调优
    model.fit(X_train_tfidf, y_train)  # 模型训练
    pre = model.predict(X_test_tfidf)  # 模型预测
    return model, X_test_tfidf, pre


def score_model():
    """ 模型评估 """
    test_data = test_dataset()
    y_test = test_data['label']
    model, X_test_tfidf, pre = nb_model()
    print(model.score(X_test_tfidf, y_test))

    print(classification_report(y_test, pre))
    print(confusion_matrix(y_test, pre))


if __name__ == '__main__':
    train_data = train_dataset()
