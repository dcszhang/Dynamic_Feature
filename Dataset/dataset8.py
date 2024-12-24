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

# 选择和打乱账户
def select_and_shuffle_accounts(accounts):
    tag1_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 1]
    tag0_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 0]
    
    # 随机选择tag为0的账户，数量是tag为1的账户数量的两倍
    double_tag1_count = random.sample(tag0_accounts, 2 * len(tag1_accounts))
    
    # 合并并打乱顺序
    selected_accounts = tag1_accounts + double_tag1_count
    random.shuffle(selected_accounts)
    
    # 返回打乱后的字典
    return dict(selected_accounts)

# 加载数据
accounts_data = load_data('transactions7.pkl')

# 选择和打乱账户
shuffled_accounts_data = select_and_shuffle_accounts(accounts_data)

# 保存数据
save_data(shuffled_accounts_data, 'transactions8.pkl')

# 打印每个账户的前十条处理后的交易记录
print("打印前十个账户的数据:")
for address, transactions in list(shuffled_accounts_data.items())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address}:")
    print(transactions)
    print("\n")

print("数据已被处理，并保存到 transactions8.pkl 中。")
