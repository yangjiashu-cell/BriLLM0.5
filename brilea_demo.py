import torch
from model import BraLM, Vocab
from tokenizers import Tokenizer
import json

# 加载官方权重
with open("./vocab_wiki_4k_en.json") as f:
    node_dict = json.load(f)
vocab = Vocab.from_node_dict(node_dict)

model = BraLM(hidden_size=32)
model.prepare_network(vocab)
model.load_state_dict(torch.load("model_en.bin", weights_only=True))
model.to("cuda")
model.eval()

bpe = Tokenizer.from_file("wiki_bpe_tokenizer_4000_bytelevel.json")

def generate_caption(condition: torch.Tensor, prompt: str = "", max_tokens=80):
    """
    condition: (1, 32) tensor from Ridge
    """
    if prompt:
        tokens = bpe.encode(prompt).tokens
        start = [vocab(tokens[i]+"->"+tokens[i+1]) for i in range(len(tokens)-1)]
    else:
        start = []
    
    ret = model.decode(start, vocab, max_new_tokens=max_tokens, condition=condition)
    decoded = [vocab.decode(p).split("->") for p in ret]
    sentence = decoded[0][0] + "".join(p[-1] for p in decoded)
    return sentence.replace("Ġ", " ")
    return sentence

# 测试1：看condition是否改变生成
cond = torch.randn(1,32).cuda() * 3.0
print(generate_caption(cond, "A photo of"))
# 你会发现每次cond不同，生成内容完全不同，证明成功