import pickle
import networkx as nx
from tqdm import tqdm
import pandas as pd
import functools
import pickle
def read_pkl(pkl_file):
    # 从pkl文件加载数据
    print(f'Reading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(data, pkl_file):
    # 将数据保存到pkl文件   
    print(f'Saving data to {pkl_file}...')
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)
def load_and_print_pkl(pkl_file):
    # 加载pkl文件
    print(f'Loading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    
    # 打印数据的前十条记录
    for i, transaction in enumerate(data):
        if i < 10:  # 仅打印前十条记录
            print(transaction)
        else:
            break
def extract_transactions(G):
    # 提取图中所有交易数据
    transactions = []
    for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True),desc=f'accounts_data_generate'):
        amount = tnx_info['amount']
        block_timestamp = int(tnx_info['timestamp'])
        tag = G.nodes[from_address]['isp']
        transaction = {
            'tag': tag,
            'from_address': from_address,
            'to_address': to_address,
            'amount': amount,
            'timestamp': block_timestamp,
        }
        transactions.append(transaction)
    return transactions

def data_generate():
    graph_file = 'MulDiGraph.pkl'
    out_file = 'transactions1.pkl'
    
    # 读取图数据
    graph = read_pkl(graph_file)
    # 提取交易数据
    transactions = extract_transactions(graph)
    # 保存交易数据到新文件
    save_pkl(transactions, out_file)

if __name__ == '__main__':
    data_generate()
    pkl_file = 'transactions1.pkl'  # 确保此文件路径正确
    load_and_print_pkl(pkl_file)

