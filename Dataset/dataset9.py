import pickle
import tqdm

# 从文件中读取数据
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 保存数据到文件
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# 删除transactions中的tag字段
def remove_tag_from_transactions(accounts):
    for address, transactions in accounts.items():
        for transaction in transactions:
            for sub_transaction in transaction['transactions']:
                if 'tag' in sub_transaction:
                    del sub_transaction['tag']

# 加载数据
accounts_data = load_data('transactions8.pkl')

# 删除tag字段
remove_tag_from_transactions(accounts_data)

# 保存数据
save_data(accounts_data, 'transactions9.pkl')

# 打印前十个账户的数据
print("打印前十个账户的数据:")
for address, transactions in list(accounts_data.items())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("tag字段已被删除，并保存到 transactions9.pkl 中。")
