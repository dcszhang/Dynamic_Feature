
import re

import numpy as np
import scipy.sparse as sp
import torch
from nltk.tokenize import TweetTokenizer
from torch.utils import data
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

"""
General functions
"""


def del_http_user_tokenize(tweet):
    space_pattern = r"\s+"
    url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"
    tweet = re.sub(space_pattern, " ", tweet)
    tweet = re.sub(url_regex, "", tweet)
    tweet = re.sub(mention_regex, "", tweet)
    return tweet


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_tweet_tokenize(string):
    tknzr = TweetTokenizer(
        reduce_len=True, preserve_case=False, strip_handles=False
    )
    tokens = tknzr.tokenize(string.lower())
    return " ".join(tokens).strip()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse):
    # coo_sparse=coo_sparse.tocoo()
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        weight.append(len(y) / (n_classes * count))
    return classes_count, weight


"""
Functions and Classes for read and organize data set
"""


class InputExample(object):


    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        self.guid = guid
        # string of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.text_a = text_a
        self.text_b = text_b
        # the label(class) for the sentence
        self.confidence = confidence
        self.label = label


class InputFeatures(object):
    def __init__(
        self,
        guid,
        tokens,
        input_ids,
        gcn_vocab_ids,
        input_mask,
        segment_ids,
        confidence,
        label_id,
    ):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence = confidence
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# def example2feature(
#     example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim
# ):
#
#     tokens_a = example.text_a.split()
#     assert example.text_b == None
#     if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
#         tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]
#
#     gcn_vocab_ids = []
#     for w in tokens_a:
#         gcn_vocab_ids.append(gcn_vocab_map.get(w, -1))
#
#     tokens = (
#         ["[CLS]"] + tokens_a + ["[SEP]" for i in range(gcn_embedding_dim + 1)]
#     )
#     segment_ids = [0] * len(tokens)
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)
#
#     feat = InputFeatures(
#         guid=example.guid,
#         tokens=tokens,
#         input_ids=input_ids,
#         gcn_vocab_ids=gcn_vocab_ids,
#         input_mask=input_mask,
#         segment_ids=segment_ids,
#         # label_id=label2idx[example.label]
#         confidence=example.confidence,
#         label_id=example.label,
#     )
#     return feat
def example2feature(example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim):
    tokens_a = example.text_a.split()  # 假设 text_a 包含的是账户地址或交易相关的字段
    assert example.text_b == None
    if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
        tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]

    gcn_vocab_ids = []
    for word in tokens_a:
        if word in gcn_vocab_map:
            gcn_vocab_ids.append(gcn_vocab_map[word])
        else:
            # 如果词汇/地址不在 gcn_vocab_map 中，使用默认值（如 -1 表示未找到）
            gcn_vocab_ids.append(gcn_vocab_map.get('UNK', -1))  # 'UNK' 可以替换为合适的默认值

    # 构建 BERT 输入的 tokens，包括 [CLS] 和 [SEP]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]" for _ in range(gcn_embedding_dim + 1)]
    segment_ids = [0] * len(tokens)

    # 将 tokens 转换为 BERT 所需的 input_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # 创建并返回 InputFeatures 对象
    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
    )
    return feat

class CorpusDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        gcn_vocab_map,
        max_seq_len,
        gcn_embedding_dim,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = gcn_vocab_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
        )

    # @classmethod
    def pad(self, batch):
        gcn_vocab_size = len(self.gcn_vocab_map)
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
        ]
        f_pad2 = lambda x, seqlen: [
            [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
            for sample in batch
        ]

        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)
        batch_gcn_vocab_ids_paded = np.array(f_pad2(5, maxlen)).reshape(-1)
        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[
            batch_gcn_vocab_ids_paded
        ][:, :-1]
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(
            len(batch), -1, gcn_vocab_size
        ).transpose(1, 2)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
        )
