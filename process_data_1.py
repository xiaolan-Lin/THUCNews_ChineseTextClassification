import pandas as pd
import re

# def process_data():
#     """
#     数据预处理
#     """
#     path = "D:\\小阔爱\\大学课程\\机器学习\\期末大作业\\sample_data"  # 小样本
#     # 调用walk方法递归遍历path目录
#     num = 0
#     str = ""
#     for root, dirs, files in os.walk(path):
#         for name in files:
#             filename = os.path.dirname(os.path.realpath(os.path.join(root, name))).split('\\')[6]
#             print(filename)
#             num += 1
#             print("第", num, "次打开文件")
#             file = open(os.path.join(root, name), encoding='utf8')
#             iter_file = iter(file)
#             i = 0
#             for line in iter_file:
#                if i == 0:
#                    str = str + filename + ','
#                str = str + line
#                i = i + 1
#                str = re.sub('\n+', '', str)
#             str = str + '\r\n'
#     move = dict.fromkeys((ord(c) for c in u"\xa0"))
#     str = str.replace(u'\u3000', u' ').replace(' ', '').translate(move)
#
#     csv_file = "D:\\小阔爱\\大学课程\\机器学习\\期末大作业\\after_sample.csv"
#     with open(csv_file, 'w', encoding='utf8') as f:
#         f.write(str)
#     print("读取完毕!")
#     print("总条数目为:", num)
#     return csv_file


# def read_dataSet(path):
#     """
#     读取数据集
#     """
#     f = open(path, encoding='utf8')
#     data = pd.read_csv(f, sep=',', header=None, names=['Label', 'Content'], error_bad_lines=False)
#     label = data['Label']
#     content = data['Content']
#     # print(data)
#     return content, label
