import pickle
from sklearn.model_selection import train_test_split

# 加载transactions10.pkl文件中的数据
with open('transactions10.pkl', 'rb') as file:
    transactions_dict = pickle.load(file)

# 将字典转换为列表，每个元素是一个"tag sentence"格式的字符串
transactions = []
for key, value_list in transactions_dict.items():
    for value in value_list:
        transactions.append(f"{value}")  # 假设key是tag，而value是描述

# 定义数据分割比例
train_size = 0.8
validation_size = 0.1
test_size = 0.1

# 先分割出训练集和剩余部分
train_data, temp_data = train_test_split(transactions, train_size=train_size, random_state=42)

# 再从剩余部分中分割出验证集和测试集
validation_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + validation_size), random_state=42)

# 保存训练集和验证集数据到TSV文件的函数
def save_to_tsv_train_dev(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("label\tsentence\n")
        for line in data:
            # 假设tag是行的开始，并将整个余下部分作为sentence
            tag, sentence = line.split(' ', 1)
            file.write(f"{tag}\t{sentence}\n")

# 保存测试集数据到TSV文件的函数
def save_to_tsv_test(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tsentence\n")
        for idx, line in enumerate(data):
            # 分割tag和后续描述
            tag, sentence = line.split(' ', 1)
            file.write(f"{idx}\t{sentence}\n")

# 保存训练集、验证集和测试集
save_to_tsv_train_dev(train_data, 'train.tsv')
save_to_tsv_train_dev(validation_data, 'dev.tsv')
save_to_tsv_test(test_data, 'test.tsv')

print("Files saved: train.tsv, dev.tsv, test.tsv")
