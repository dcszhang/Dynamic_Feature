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

# 转换交易数据为描述性文本
def convert_transactions_to_text(accounts):
    for address, transactions in accounts.items():
        for idx, transaction in enumerate(transactions):
            tag = transaction['tag']
            transaction_descriptions = []
            for sub_transaction in transaction['transactions']:
                # 构建单个交易的描述
                description = ' '.join([f"{key}: {sub_transaction[key]}" for key in sub_transaction])
                transaction_descriptions.append(description)
            # 更新交易数据为一行文本描述
            transactions[idx] = f"{tag} {'  '.join(transaction_descriptions)}."

# 加载数据
accounts_data = load_data('transactions9.pkl')

# 转换交易数据为文本描述
convert_transactions_to_text(accounts_data)

# 保存数据
save_data(accounts_data, 'transactions10.pkl')

# 打印前十个账户的数据
print("打印前十个账户的数据:")
for address, transactions in list(accounts_data.items())[:10]:  # 只展示前十个账户的数据
    print(f"账户 {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("数据已被转换为描述性文本并保存到 transactions10.pkl 中。")
