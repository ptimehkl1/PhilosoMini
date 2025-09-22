import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import struct  # 浮点数到十六进制的转换


# 定义之前的PhilosoMini模型和训练/数据准备函数
# (为了代码的完整性，这里再次包含它们，实际运行时可以从之前的代码导入)

class PhilosoMini(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(PhilosoMini, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.predictor = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        logits = self.predictor(embeddings)
        return logits

    def get_word_meaning(self, word_id):
        with torch.no_grad():
            return self.embedding(torch.tensor([word_id])).squeeze().numpy()

    def analyze_transition_preferences(self, word_id):
        with torch.no_grad():
            logits = self.forward(torch.tensor([word_id]))
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            return probs


def create_philosophical_dataset():
    vocab = {
        '<start>': 0, '<end>': 1,
        '小猫': 2, '小狗': 3, '鸟儿': 4,
        '跑步': 5, '飞翔': 6, '睡觉': 7,
        '快乐': 8, '安静': 9,
        '在': 10, '花园': 11, '天空': 12,
        '里': 13, '中': 14
    }
    id_to_word = {v: k for k, v in vocab.items()}
    training_sentences = [
        "小猫 跑步", "小狗 跑步", "鸟儿 飞翔",
        "小猫 睡觉", "小狗 睡觉",
        "小猫 在 花园 里", "鸟儿 在 天空 中",
        "小猫 快乐", "小狗 快乐", "鸟儿 安静"
    ]
    return vocab, id_to_word, training_sentences


def prepare_training_data(sentences, vocab):
    training_pairs = []
    for sentence in sentences:
        words = ['<start>'] + sentence.split() + ['<end>']
        word_ids = [vocab[word] for word in words]
        for i in range(len(word_ids) - 1):
            current_word = word_ids[i]
            next_word = word_ids[i + 1]
            training_pairs.append((current_word, next_word))
    return training_pairs


def train_model(model, training_pairs, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for current_word, next_word in training_pairs:
            optimizer.zero_grad()
            current_tensor = torch.tensor([current_word])
            target_tensor = torch.tensor([next_word])
            logits = model(current_tensor)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model  # 返回训练后的模型


# 辅助函数：将float32转换为十六进制表示
def float_to_hex(f):
    # struct.pack('<f', f) 将float f打包成小端序的4字节二进制数据
    # struct.unpack('<I', ...) 将这4字节二进制数据解包成无符号整数
    # hex(...) 将无符号整数转换为十六进制字符串
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


# --- 运行训练并展示参数 ---

print("🌟 PhilosoMini模型参数的十六进制展示与含义解析 🌟")
print("=" * 60)

# 1. 准备数据
vocab, id_to_word, sentences = create_philosophical_dataset()
training_pairs = prepare_training_data(sentences, vocab)

# 2. 创建并训练模型
model = PhilosoMini(vocab_size=len(vocab), embed_dim=4)
print("模型开始训练...")
model = train_model(model, training_pairs, epochs=200)  # 增加训练轮数，让参数更稳定
print("模型训练完成！\n")

print(f"📊 模型总参数量: {sum(p.numel() for p in model.parameters())}个\n")

# 3. 遍历并展示每个参数
for name, param in model.named_parameters():
    print(f"--- 参数组: {name} --- (形状: {param.shape})")

    if "embedding.weight" in name:
        print("💡 含义: 这些参数定义了每个词汇在'意义空间'中的位置（坐标）。")
        print("   每个词被4个浮点数（4个维度）描述。")
        print("   例如，'小猫'的这4个数字，就代表了'小猫'的某种数字化的'性格'或'属性'。\n")

        for i in range(param.shape[0]):  # 遍历每个词
            word = id_to_word[i]
            print(f"   词汇 '{word}' (ID:{i}) 的意义向量:")
            for j in range(param.shape[1]):  # 遍历每个维度
                value = param[i, j].item()
                hex_val = float_to_hex(value)
                print(f"     维度 {j + 1}: 十进制={value:.6f}, 十六进制={hex_val}")
            print()

    elif "predictor.weight" in name:
        print("💡 含义: 这些参数定义了词汇之间的'转移规则'或'引力强度'。")
        print("   它们决定了当模型看到一个词时，它的'意义向量'会如何影响下一个词出现的可能性。")
        print(
            "   例如，如果'小猫'的'活泼'维度很高，并且'跑步'的某个权重对'活泼'维度很敏感，那么'小猫'后面就更容易出现'跑步'。\n")

        for i in range(param.shape[0]):  # 遍历每个可能的下一个词
            next_word = id_to_word[i]
            print(f"   预测下一个词为 '{next_word}' (ID:{i}) 的权重（受前一个词的4个维度影响）:")
            for j in range(param.shape[1]):  # 遍历前一个词的每个维度
                value = param[i, j].item()
                hex_val = float_to_hex(value)
                print(f"     影响来自前一个词的维度 {j + 1}: 十进制={value:.6f}, 十六进制={hex_val}")
            print()

    elif "predictor.bias" in name:
        print("💡 含义: 这些参数定义了每个词汇的'基础偏好'或'默认倾向'。")
        print("   它们是模型在没有任何前文信息时，对某个词出现的'初始分数'。")
        print("   例如，如果'跑步'的偏置很高，那么它就比偏置低的词更容易被选作下一个词，即使前一个词对它的影响不大。\n")

        for i in range(param.shape[0]):  # 遍历每个词
            word = id_to_word[i]
            value = param[i].item()
            hex_val = float_to_hex(value)
            print(f"   词汇 '{word}' (ID:{i}) 的基础偏置: 十进制={value:.6f}, 十六进制={hex_val}")
        print()

print("=" * 60)
print("✨ 总结：")
print("这些数字，无论是十进制还是十六进制，都是PhiloMini'大脑'的全部知识。")
print("它们共同决定了模型如何理解词汇、如何推理词汇之间的关系，最终生成新的文本。")
print("虽然十六进制看起来神秘，但它只是浮点数在计算机中的一种存储形式。")
print("真正有意义的是这些浮点数的大小和它们在模型结构中的位置和作用。")
print("就像基因的ATCG序列本身没有意义，但它们编码的蛋白质和生命功能才有意义一样。")
print("AI的智能，就藏在这些看似无序的数字排列中，通过精妙的数学运算，涌现出理解和创造的能力。")
