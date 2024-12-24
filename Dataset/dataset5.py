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

# 删除特定字段
def remove_fields(accounts, fields):
    for address in tqdm.tqdm(accounts.keys(), desc="删除字段"):
        for transaction in accounts[address]:
            for field in fields:
                if field in transaction:
                    del transaction[field]

# 加载数据
accounts_data = load_data('transactions4.pkl')

# 需要删除的字段
fields_to_remove = ['from_address', 'to_address', 'timestamp']

# 删除字段
remove_fields(accounts_data, fields_to_remove)

# 保存数据
save_data(accounts_data, 'transactions5.pkl')

# 打印每个账户的前十条处理后的交易记录
print("打印每个账户的前十条处理后的交易记录:")
for address in list(accounts_data.keys())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address} 的前十条交易记录:")
    for transaction in accounts_data[address][:10]:  # 每个账户显示前十条记录
        print(transaction)
    print("\n")

print("字段已删除，并保存数据到 transactions5.pkl 中。")
