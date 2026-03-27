from datasets import load_dataset

# 加载 COCO 2017 训练集（"2017" 可替换为 "2014" 等其他版本）
coco_train = load_dataset("coco", "2017", split="train")
# 加载 COCO 2017 验证集
coco_val = load_dataset("coco", "2017", split="validation")

# 验证加载成功（查看前1条数据）
print(coco_train[0])