import math
import inspect
import torch
import torch.nn as nn
import torch.nn.init as init

# for huggingface transformers 0.6.2;
from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)


class VocabGraphConvolution(nn.Module):
    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if (
                    n.startswith("W")
                    or n.startswith("a")
                    or n in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            # H_vh = vocab_adj_list[i].mm(getattr(self, "W%d_vh" % i))
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")
            H_vh = torch.sparse.mm(vocab_adj_list[i].float(), getattr(self, "W%d_vh" % i))

            # H_vh=self.dropout(F.elu(H_vh))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, "W%d_vh" % i))
                H_linear = self.dropout(H_linear)
                H_dh += H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out = self.fc_hc(fused_H)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    实现 DiffSoftmax，用于在训练中使用软标签或硬标签。
    - tau: 温度参数，控制 softmax 输出的平滑度
    - hard: 是否使用硬标签
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class DynamicFusionLayer(nn.Module):
    def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
        super(DynamicFusionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 3),
            # nn.Softmax(dim=-1),
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_embeddings, gcn_enhanced_embeddings):
        concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)

        gate_logits = self.gate_network(concat_embeddings)
        gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

        gate_bert_only = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn_enhanced = gate_values[:, :, 1].unsqueeze(-1)
        gate_gcn_bert_weighted = gate_values[:, :, 2].unsqueeze(-1)

        embeddings_bert_only = bert_embeddings
        embeddings_gcn_enhanced = gcn_enhanced_embeddings
        embeddings_gcn_bert_weighted = self.fusion_weight * bert_embeddings + (1 - self.fusion_weight) * gcn_enhanced_embeddings

        fused_embeddings = (
                gate_bert_only * embeddings_bert_only +
                gate_gcn_enhanced * embeddings_gcn_enhanced +
                gate_gcn_bert_weighted * embeddings_gcn_bert_weighted
        )

        return fused_embeddings


class ETH_GBertEmbeddings(BertEmbeddings):
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super(ETH_GBertEmbeddings, self).__init__(config)
        assert gcn_embedding_dim >= 0
        self.gcn_embedding_dim = gcn_embedding_dim
        self.vocab_gcn = VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim)

        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_embeddings(input_ids)

        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
                       ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        new_words_embeddings = self.dynamic_fusion_layer(words_embeddings, gcn_words_embeddings)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = new_words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ETH_GBertModel(BertModel):
    def __init__(
            self,
            config,
            gcn_adj_dim,
            gcn_adj_num,
            gcn_embedding_dim,
            num_labels,
            output_attentions=False,
            keep_multihead_output=False,
    ):
        super().__init__(config)
        self.embeddings = ETH_GBertEmbeddings(
            config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim
        )
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.output_attentions = config.output_attentions if hasattr(config, 'output_attentions') else False
        self.keep_multihead_output = config.keep_multihead_output if hasattr(config, 'keep_multihead_output') else False
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.apply(self.init_bert_weights)

    def forward(
            self,
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            output_all_encoded_layers=False,
            head_mask=None,
    ):
        vocab_adj_list = [adj * 0 for adj in vocab_adj_list]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids,
            attention_mask,
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                head_mask = head_mask.expand_as(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:

            head_mask = [None] * self.config.num_hidden_layers

        encoder_args = {

        }
        if 'head_mask' in inspect.signature(self.encoder.forward).parameters:
            encoder_args['head_mask'] = head_mask

        if self.output_attentions:
            output_all_encoded_layers = True

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args
            # head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits
