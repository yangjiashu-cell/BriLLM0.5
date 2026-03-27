import json
from collections import defaultdict
import torch
import torch.nn as nn
from model import BraLM, Vocab
from tokenizers import Tokenizer

bpe_tokenizer = Tokenizer.from_file("wiki_bpe_tokenizer_4000_bytelevel.json")

# 加载部分（同阶段1简化）
with open('word_frequency.json', 'r') as f:
    freq_dict = json.load(f)
zero_freq_edges = defaultdict(list)
for s in freq_dict:
    zero_freq_edges[s] = [t for t in freq_dict[s] if freq_dict[s][t] == 0]
with open("vocab_wiki_4k_en.json") as f:
    node_dict = json.load(f)
for s in node_dict:
    if s not in zero_freq_edges:
        zero_freq_edges[s] = []
vocab = Vocab.from_node_dict(node_dict)
model = BraLM(hidden_size=32, zero_freq_edges=zero_freq_edges, vocab=vocab)
model.prepare_network(vocab)
state_dict = torch.load("model_en.bin", weights_only=True)
#model.load_state_dict(state_dict)
# 忽略缺失的 cond_proj 参数（因为老权重里没有）
model.load_state_dict(state_dict, strict=False)

# 手动初始化我们新加的 cond_proj（非常重要！）
if not hasattr(model, 'cond_proj') or model.cond_proj is None:
    model.cond_proj = nn.Linear(32, model.hidden_size).to(model.device)
    nn.init.xavier_uniform_(model.cond_proj.weight)
    nn.init.zeros_(model.cond_proj.bias)
model.to_device("cuda")

def gen(prompt="", cond=None, max_new_tokens=40, temperature=0.7):
    tokens = bpe_tokenizer.encode(prompt).tokens
    start = [vocab(tokens[i] + '->' + tokens[i+1]) for i in range(len(tokens)-1)] if len(tokens)>1 else []
    ret = model.decode(start, vocab, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.5, condition=cond)  # greedy (False) + low temp
    #ret = model.decode(start, vocab, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, condition=cond)
    s = "".join([vocab.decode(p).split("->")[0] if i==0 else vocab.decode(p).split("->")[1] for i,p in enumerate(ret)]).replace("Ġ", " ")
    return s

