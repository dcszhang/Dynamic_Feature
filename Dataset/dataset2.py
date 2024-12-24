import pickle

# 从文件中读取数据
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 保存数据到文件
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# 主处理函数
def process_transactions(transactions):
    # 创建一个字典来存储每个地址的交易
    accounts = {}

    # 处理交易数据
    for tx in transactions:
        # 为“转出”交易添加
        from_address = tx['from_address']
        if from_address not in accounts:
            accounts[from_address] = []
        accounts[from_address].append({**tx, 'in_out': 1})  # 添加转出标志

        # 为“转入”交易添加
        to_address = tx['to_address']
        if to_address not in accounts:
            accounts[to_address] = []
        accounts[to_address].append({**tx, 'in_out': 0})  # 添加转入标志

    return accounts

# 加载数据
transactions = load_data('transactions1.pkl')

# 处理数据
processed_data = process_transactions(transactions)

# 保存数据
save_data(processed_data, 'transactions2.pkl')

# 打印前十行数据进行检查
for address in list(processed_data.keys())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address} 的交易记录:")
    for transaction in processed_data[address][:5]:  # 每个账户显示前五条记录
        print(transaction)
    print("\n")
