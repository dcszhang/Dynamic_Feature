import pickle

# 从文件中读取数据
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 保存数据到文件
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# 对每个账户的交易数据按时间戳排序
def sort_transactions_by_timestamp(accounts):
    sorted_accounts = {}
    for address, transactions in accounts.items():
        sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
    return sorted_accounts

# 加载数据
accounts_data = load_data('transactions2.pkl')

# 排序数据
sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

# 打印每个账户的前十条排序后的交易记录
print("打印每个账户的前十条排序后的交易记录:")
for address in list(sorted_accounts_data.keys())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address} 的前十条交易记录:")
    for transaction in sorted_accounts_data[address][:10]:  # 每个账户显示前十条记录
        print(transaction)
    print("\n")

# 保存数据
save_data(sorted_accounts_data, 'transactions3.pkl')

print("数据已经按每个账户的时间戳进行排序，并保存到 transactions3.pkl 中。")
