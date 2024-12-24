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

# 删除除第一条外的所有交易的tag字段
def remove_tag_except_first(accounts):
    for address, transactions in accounts.items():
        for i in range(1, len(transactions)):
            if 'tag' in transactions[i]:
                del transactions[i]['tag']

# 合并每个账户的所有交易数据为一项
def merge_transactions(accounts):
    for address in accounts.keys():
        if accounts[address]:
            first_tag = accounts[address][0]['tag']  # 保留第一条交易的tag
            merged_data = {'tag': first_tag, 'transactions': accounts[address]}
            accounts[address] = [merged_data]

# 加载数据
accounts_data = load_data('transactions6.pkl')

# 删除tag字段
remove_tag_except_first(accounts_data)

# 合并交易数据
merge_transactions(accounts_data)

# 保存数据
save_data(accounts_data, 'transactions7.pkl')

# 打印每个账户的前十条处理后的交易记录
print("打印每个账户的前十条处理后的交易记录:")
for address in list(accounts_data.keys())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address} 的前十条交易记录:")
    for transaction in accounts_data[address][:10]:  # 每个账户显示前十条记录
        print(transaction)
    print("\n")

print("交易数据已被处理，并保存到 transactions7.pkl 中。")
