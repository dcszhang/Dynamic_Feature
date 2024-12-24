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

# 添加n-gram数据到每个交易
def add_n_grams(accounts):
    for address, transactions in tqdm.tqdm(accounts.items(), desc="处理n-gram数据"):
        for n in range(2, 6):  # 处理2-gram到5-gram
            gram_key = f"{n}-gram"
            for i in range(len(transactions)):
                if i < n-1:
                    transactions[i][gram_key] = 0  # n-1 个初始值设置为0
                else:
                    transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']

# 加载数据
accounts_data = load_data('transactions3.pkl')

# 添加n-gram数据
add_n_grams(accounts_data)

# 保存数据
save_data(accounts_data, 'transactions4.pkl')

# 打印每个账户的前十条处理后的交易记录
print("打印每个账户的前十条处理后的交易记录:")
for address in list(accounts_data.keys())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address} 的前十条交易记录:")
    for transaction in accounts_data[address][:10]:  # 每个账户显示前十条记录
        print(transaction)
    print("\n")

print("数据已经按照n-gram计算完毕，并保存到 transactions4.pkl 中。")
