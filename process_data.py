import os


"""
31w数据集
如果当数据量不是很大的时候（万级别以下）的时候将训练集、验证集以及测试集划分为6：2：2；
若是数据很大，可以将训练集、验证集、测试集比例调整为98：1：1；
但是当可用的数据很少的情况下也可以使用一些高级的方法，比如留出方，K折交叉验证等。
训练集：训练模型
验证集：评估模型
测试集：测试模型
"""


def read_file(filename):
    """
    读取文件并将文件格式化转换为一行
    """
    with open(filename, 'r', encoding='utf8') as f:
        return f.read().replace('\t', '').replace('\n', '').replace('\u3000', '')


def save_file(dirname):
    """
    dirname: 源文件目录
    将多个文件整理合并至三个文件中：训练集（train_data）、测试集（test_data）、验证集（val_data）
    文件内容格式：类别\t内容
    """
    train_file = open(r"D:\PycharmProjects\THUCNews_after_data\train_data.txt", 'w', encoding='utf8')
    test_file = open(r"D:\PycharmProjects\THUCNews_after_data\test_data.txt", 'w', encoding='utf8')
    # val_data = open(r"D:\PycharmProjects\THUCNews_after_data\val_data.txt", 'w', encoding='utf8')
    for category in os.listdir(dirname):
        category_dir = os.path.join(dirname, category)
        print(category)
        print(category_dir)
        # if not os.path.isdir(category_dir):
        #     continue
        files = os.listdir(category_dir)
        data_length = len(files)
        count = 0
        for f in files:
            filename = os.path.join(category_dir, f)
            content = read_file(filename)
            print("数据处理完毕！")
            if count < int(data_length/10)*8:
                train_file.write(category + '\t' + content + '\n')
            else:
                test_file.write(category + '\t' + content + '\n')
            # if count < 60:
            #     train_file.write(category + '\t' + content + '\n')
            # elif count < 80:
            #     test_file.write(category + '\t' + content + '\n')
            # else:
            #     val_data.write(category + '\t' + content + '\n')
            count += 1

        print("结束数据集拆分")
    train_file.close()
    test_file.close()
    # val_data.close()


if __name__ == '__main__':
    # dirname = r"D:\小阔爱\大学课程\机器学习\期末大作业\sample_data"
    # save_file(dirname)
    # print(len(open(r"D:\PycharmProjects\THUCNews_after_data\train_data.txt", 'r', encoding='utf8').readlines()))
    # print(len(open(r"D:\PycharmProjects\THUCNews_after_data\test_data.txt", 'r', encoding='utf8').readlines()))
    # print(len(open(r"D:\PycharmProjects\THUCNews_after_data\val_data.txt", 'r', encoding='utf8').readlines()))
    import pandas as pd
    data = pd.read_csv(r"D:\PycharmProjects\THUCNews_after_data\test_data.txt", sep='\t', encoding='utf8',
                       names=['label', 'content'])
    # print(len(data[data['label'] == '体育']))  # 105280 26324
    # print(len(data[data['label'] == '时政']))  # 50464  12622
    # print(len(data[data['label'] == '星座']))  # 2856  722
    # print(len(data[data['label'] == '娱乐']))  # 74104  18528
    # print(len(data[data['label'] == '游戏']))  # 19496  4877
    # print(data[data['label'] == '体育'])
    # print(data)


# 时政 63085
# 体育 131604
# 娱乐 92631
# 游戏 24372
# 星座 3577


