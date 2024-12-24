import pickle
import random
import numpy as np

# 从文件中读取数据
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 保存数据到文件
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# 分类统计正常和异常账户，并加载交易数据
data_filename = 'transactions4.pkl'
accounts_data = load_data(data_filename)

normal_accounts = {}
abnormal_accounts = {}

for address, transactions in accounts_data.items():
    if transactions[0]['tag'] == 0:
        normal_accounts[address] = transactions
    elif transactions[0]['tag'] == 1:
        abnormal_accounts[address] = transactions

# 获取异常账户数量
num_abnormal = len(abnormal_accounts)

# 从正常账户中随机选择异常账户数目的两倍
selected_normal_accounts = random.sample(normal_accounts.keys(), 2 * num_abnormal)
adjusted_normal_accounts = {addr: normal_accounts[addr] for addr in selected_normal_accounts}

# 合并调整后的正常账户和所有异常账户
adjusted_accounts_data = {**adjusted_normal_accounts, **abnormal_accounts}

# 保存调整后的数据
save_data_filename = 'adjusted_transactions4.pkl'
save_data(adjusted_accounts_data, save_data_filename)

print(f"数据已调整并保存到 {save_data_filename}")
print(f"异常账户数: {len(abnormal_accounts)}")
print(f"选中的正常账户数: {len(adjusted_normal_accounts)}")

# 打印前十个账户的前十条交易记录
print("\n前十个账户的前十条交易记录:")
for address in list(adjusted_accounts_data.keys())[:10]:  # 只展示前十个账户的数据
    print(f"\n账户 {address} 的前十条交易记录:")
    for transaction in adjusted_accounts_data[address][:10]:  # 每个账户显示前十条记录
        print(transaction)

# 定义权重计算函数
def calculate_weight(transaction):
    weights = []
    if '2-gram' in transaction:
        weights.append(transaction['2-gram'] * 0.1)
    if '3-gram' in transaction:
        weights.append(transaction['3-gram'] * 0.2)
    if '4-gram' in transaction:
        weights.append(transaction['4-gram'] * 0.3)
    if '5-gram' in transaction:
        weights.append(transaction['5-gram'] * 0.4)
    return np.sum(weights) if weights else 0  # 计算平均值，如果列表为空则返回0

# 提取所有独特的账户地址，只包含当前剩余账户
addresses = set(adjusted_accounts_data.keys())

# 地址到索引的映射
address_to_index = {addr: idx for idx, addr in enumerate(addresses)}

# 创建邻接矩阵
n = len(addresses)
adj_matrix = np.zeros((n, n), dtype=float)  # 使用float类型以存储权重
# 保存地址到索引的映射
save_data(address_to_index, 'data_Dataset.address_to_index')
# 填充邻接矩阵
for account, transactions in adjusted_accounts_data.items():
    for transaction in transactions:
        from_addr = transaction['from_address']
        to_addr = transaction['to_address']
        if from_addr in addresses and to_addr in addresses:
            from_idx = address_to_index[from_addr]
            to_idx = address_to_index[to_addr]
            weight = calculate_weight(transaction)  # 计算权重
            adj_matrix[from_idx, to_idx] += weight  # 累加权重

# 保存邻接矩阵
save_data(adj_matrix, 'weighted_adjacency_matrix.pkl')
