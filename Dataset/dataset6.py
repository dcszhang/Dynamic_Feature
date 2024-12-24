import pickle
import random
import tqdm

# 从文件中读取数据
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 保存数据到文件
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# 打乱每个账户中的交易数据顺序
def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys(), desc="打乱交易顺序"):
        random.shuffle(accounts[address])

# 加载数据
accounts_data = load_data('transactions5.pkl')

# 打乱交易数据
shuffle_transactions(accounts_data)

# 保存数据
save_data(accounts_data, 'transactions6.pkl')

# 打印每个账户的前十条处理后的交易记录
print("打印每个账户的前十条处理后的交易记录:")
for address in list(accounts_data.keys())[:5]:  # 只展示前十个账户的数据
    print(f"账户 {address} 的前十条交易记录:")
    for transaction in accounts_data[address][:5]:  # 每个账户显示前十条记录
        print(transaction)
    print("\n")

print("交易数据已被打乱，并保存到 transactions6.pkl 中。")
