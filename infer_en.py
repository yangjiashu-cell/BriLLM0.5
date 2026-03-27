import json
from collections import defaultdict
import torch
from model import BraLM, Vocab
from tokenizers import Tokenizer

bpe_tokenizer = Tokenizer.from_file("wiki_bpe_tokenizer_4000_bytelevel.json")

def decode_en_sentence(head, max_token=32, do_sample=False):
    bpe_tokens = bpe_tokenizer.encode(head).tokens
    if len(bpe_tokens) < 2:
        return head
    start = [vocab(bpe_tokens[i] + '->' + bpe_tokens[i+1]) for i in range(len(bpe_tokens)-1)]
    ret = model.decode(start, vocab, max_token, do_sample)
    decode_tuple_list = [vocab.decode(p).split('->') for p in ret]
    decode_sentence = decode_tuple_list[0][0] + "".join([p[-1] for p in decode_tuple_list])
    return decode_sentence

# 加载frequency构建zero_freq_edges
with open('word_frequency.json', 'r') as f:
    freq_dict = json.load(f)

zero_freq_edges = defaultdict(list)
for s in freq_dict:
    zero_freq_edges[s] = [t for t in freq_dict[s] if freq_dict[s][t] == 0]

# 确保vocab所有s有默认
with open("vocab_wiki_4k_en.json") as f:
    node_dict = json.load(f)
for s in node_dict:
    if s not in zero_freq_edges:
        zero_freq_edges[s] = []

vocab = Vocab.from_node_dict(node_dict)

model = BraLM(hidden_size=32, zero_freq_edges=zero_freq_edges, vocab=vocab)
model.prepare_network(vocab)

state_dict = torch.load("model_en.bin", weights_only=True)
model.load_state_dict(state_dict)
model.to_device("cuda")

head = "In frogs, the hind legs are larger"
encoding = bpe_tokenizer.encode(head)
token_len = len(encoding.ids)
max_token = 32 - token_len
decode_sentence = decode_en_sentence(head, max_token).replace("Ġ", " ")

print(decode_sentence)