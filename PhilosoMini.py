import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("🧠 PhilosoMini: 探索智能的最小完整单元\n")
print("=" * 60)


class PhilosoMini(nn.Module):
    """
    极简语言模型：用最少参数展现语言生成的本质

    哲学思考：
    - 每个词的嵌入向量代表其在"意义空间"中的坐标
    - 线性层学习词与词之间的转移概率
    - 这是对人类语言直觉的数学抽象
    """

    def __init__(self, vocab_size, embed_dim):
        super(PhilosoMini, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # 词嵌入：将离散符号映射到连续意义空间
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 预测层：从当前词的意义预测下一词的概率分布
        self.predictor = nn.Linear(embed_dim, vocab_size)

        # 计算参数总数
        total_params = sum(p.numel() for p in self.parameters())

        print(f"🔢 模型参数统计:")
        print(f"   词汇量: {vocab_size}")
        print(f"   嵌入维度: {embed_dim}")
        print(f"   嵌入层参数: {vocab_size * embed_dim}")
        print(f"   预测层参数: {embed_dim * vocab_size + vocab_size}")
        print(f"   总参数量: {total_params}")
        print(f"   (相当于人脑神经连接的10^-12分之一)\n")

    def forward(self, input_ids):
        """
        前向传播：思维的数学化过程
        input_ids: [batch_size] 当前词的ID
        返回: [batch_size, vocab_size] 下一词的概率logits
        """
        # 步骤1: 符号到意义的转换
        embeddings = self.embedding(input_ids)  # [batch, embed_dim]

        # 步骤2: 基于当前意义预测下一词
        logits = self.predictor(embeddings)  # [batch, vocab_size]

        return logits

    def get_word_meaning(self, word_id):
        """获取词汇的'意义向量'（用于哲学分析）"""
        with torch.no_grad():
            return self.embedding(torch.tensor([word_id])).squeeze().numpy()

    def analyze_transition_preferences(self, word_id):
        """分析给定词对下一词的偏好（展现学习到的语言规律）"""
        with torch.no_grad():
            logits = self.forward(torch.tensor([word_id]))
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            return probs


def create_philosophical_dataset():
    """
    创建体现语言基本模式的训练数据

    设计哲学：选择最能体现语言规律的词汇和句式
    """

    # 精心选择的15个词汇：涵盖主体、动作、对象、修饰
    vocab = {
        '<start>': 0, '<end>': 1,  # 边界标记
        '小猫': 2, '小狗': 3, '鸟儿': 4,  # 主体
        '跑步': 5, '飞翔': 6, '睡觉': 7,  # 动作
        '快乐': 8, '安静': 9,  # 状态
        '在': 10, '花园': 11, '天空': 12,  # 场所
        '里': 13, '中': 14  # 介词
    }

    id_to_word = {v: k for k, v in vocab.items()}

    # 训练语料：体现基本语言模式
    training_sentences = [
        "小猫 跑步",
        "小狗 跑步",
        "鸟儿 飞翔",
        "小猫 睡觉",
        "小狗 睡觉",
        "小猫 在 花园 里",
        "鸟儿 在 天空 中",
        "小猫 快乐",
        "小狗 快乐",
        "鸟儿 安静"
    ]

    print("📖 哲学训练语料:")
    print(f"   词汇表大小: {len(vocab)}")
    print(f"   训练句子数: {len(training_sentences)}")
    print("   示例句子:")
    for i, sent in enumerate(training_sentences[:3]):
        print(f"     {i + 1}. {sent}")
    print("   ...(更多句子体现动物-动作-场所的基本模式)\n")

    return vocab, id_to_word, training_sentences


def prepare_training_data(sentences, vocab):
    """将文本转换为模型训练数据"""

    training_pairs = []

    for sentence in sentences:
        words = ['<start>'] + sentence.split() + ['<end>']
        word_ids = [vocab[word] for word in words]

        # 创建(当前词, 下一词)训练对
        for i in range(len(word_ids) - 1):
            current_word = word_ids[i]
            next_word = word_ids[i + 1]
            training_pairs.append((current_word, next_word))

    return training_pairs


def train_model(model, training_pairs, epochs=100):
    """
    训练过程：AI学习语言模式的过程

    哲学意义：通过最小化预测错误，模型逐渐内化语言规律
    """

    print("🎓 开始训练过程:")
    print(f"   训练轮数: {epochs}")
    print(f"   学习任务: 给定当前词，预测最可能的下一词")
    print(f"   训练样本数: {len(training_pairs)}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for current_word, next_word in training_pairs:
            optimizer.zero_grad()

            # 前向传播
            current_tensor = torch.tensor([current_word])
            target_tensor = torch.tensor([next_word])

            logits = model(current_tensor)
            loss = criterion(logits, target_tensor)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_pairs)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"   第{epoch + 1:3d}轮: 平均损失 = {avg_loss:.4f}")

    print("   🎉 训练完成！模型已学会基本的语言转移模式\n")
    return losses


def analyze_learned_patterns(model, vocab, id_to_word):
    """
    分析模型学到的语言模式

    哲学价值：让我们看到AI如何理解和编码语言规律
    """

    print("🔍 分析模型学到的语言模式:")

    # 分析关键词的转移偏好
    key_words = ['<start>', '小猫', '小狗', '鸟儿', '在']

    for word in key_words:
        if word in vocab:
            word_id = vocab[word]
            probs = model.analyze_transition_preferences(word_id)

            # 找出概率最高的3个后续词
            top_indices = np.argsort(probs)[-3:][::-1]

            print(f"   '{word}' 之后最可能的词:")
            for i, idx in enumerate(top_indices):
                next_word = id_to_word[idx]
                prob = probs[idx]
                print(f"     {i + 1}. '{next_word}' (概率: {prob:.3f})")
            print()


def demonstrate_meaning_space(model, vocab, id_to_word):
    """
    演示词汇的意义空间

    哲学启发：展示AI如何将符号转换为意义的数学表示
    """

    print("🌟 探索词汇的意义空间:")

    # 获取所有词汇的嵌入向量
    word_meanings = {}
    for word, word_id in vocab.items():
        if word not in ['<start>', '<end>']:
            meaning_vector = model.get_word_meaning(word_id)
            word_meanings[word] = meaning_vector

    # 计算词汇间的相似性
    print("   词汇相似性分析（余弦相似度）:")

    words = list(word_meanings.keys())
    similarities = []

    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j:
                vec1 = word_meanings[word1]
                vec2 = word_meanings[word2]

                # 计算余弦相似度
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities.append((word1, word2, cosine_sim))

    # 显示最相似的词对
    similarities.sort(key=lambda x: x[2], reverse=True)
    print("   最相似的词对:")
    for word1, word2, sim in similarities[:5]:
        print(f"     '{word1}' ↔ '{word2}': {sim:.3f}")

    print()


def generate_text(model, vocab, id_to_word, start_word='<start>',
                  max_length=8, temperature=1.0, show_process=True):
    """
    文本生成：AI的创造过程

    哲学思考：这是确定性数学与随机性创造的结合 
    """

    if show_process:
        print(f"🎨 文本生成过程:")
        print(f"   起始词: '{start_word}'")
        print(f"   最大长度: {max_length}")
        print(f"   创造性温度: {temperature}")
        print()

    model.eval()

    if start_word not in vocab:
        start_word = '<start>'

    current_id = vocab[start_word]
    generated_words = [start_word]

    with torch.no_grad():
        for step in range(max_length - 1):
            # 获取下一词的概率分布
            logits = model(torch.tensor([current_id]))

            # 应用温度调节创造性
            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=1).squeeze()

            # 概率采样（体现创造性）
            if temperature > 0:
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = torch.argmax(probs).item()

            next_word = id_to_word[next_id]

            if show_process and step < 3:
                top_probs, top_indices = torch.topk(probs, 3)
                print(f"   步骤{step + 1}: 选择 '{next_word}' (概率: {probs[next_id]:.3f})")
                candidates = [id_to_word[idx.item()] for idx in top_indices]
                candidate_probs = [f"{p:.3f}" for p in top_probs]
                print(f"     候选: {candidates}")
                print(f"     概率: {candidate_probs}")

            generated_words.append(next_word)

            if next_word == '<end>':
                break

            current_id = next_id

    generated_text = ' '.join([w for w in generated_words if w not in ['<start>', '<end>']])

    if show_process:
        print(f"   ✨ 生成结果: '{generated_text}'\n")

    return generated_text


def philosophical_reflection(model, vocab):
    """
    哲学反思：从极简模型看智能本质
    """

    print("=" * 60)
    print("🤔 哲学反思：智能的本质特征")
    print("=" * 60)

    param_count = sum(p.numel() for p in model.parameters())

    print(f"💭 我们观察到了什么？")
    print()
    print(f"1. 【参数的哲学意义】")
    print(f"   仅用{param_count}个数字，模型就能展现基本的语言能力")
    print(f"   这些参数编码了词汇的'意义'和词间的'关系'")
    print(f"   意义不再是抽象概念，而是具体的数学向量")
    print()

    print(f"2. 【学习的本质】")
    print(f"   训练过程就是不断调整这些数字")
    print(f"   每次调整都让模型更好地预测'下一个词'")
    print(f"   这类似于人类通过经验修正直觉的过程")
    print()

    print(f"3. 【创造性的数学基础】")
    print(f"   温度参数控制随机性与确定性的平衡")
    print(f"   创造性可能就是'有约束的随机性'")
    print(f"   AI的'想象力'源于概率分布的采样")
    print()

    print(f"4. 【涌现现象】")
    print(f"   简单的数学运算重复应用产生了复杂行为")
    print(f"   智能可能是复杂性的涌现特性")
    print(f"   从量变到质变的哲学原理在此体现")
    print()

    print(f"5. 【规模效应的启示】")
    print(f"   如果{param_count}个参数能产生基本智能...")
    print(f"   那么万亿参数的模型会展现什么能力？")
    print(f"   这让我们思考智能的边界和可能性")
    print()


def main():
    """
    主函数：完整的哲学实验
    """

    print("🔬 PhilosoMini实验：用最少参数理解语言智能")
    print("探索问题：智能的最小完整单元是什么？\n")

    # 1. 准备数据
    vocab, id_to_word, sentences = create_philosophical_dataset()
    training_pairs = prepare_training_data(sentences, vocab)

    # 2. 创建模型
    model = PhilosoMini(vocab_size=len(vocab), embed_dim=4)

    # 3. 训练前生成（展示随机状态）
    print("🍼 训练前的生成（随机状态）:")
    untrained_text = generate_text(model, vocab, id_to_word,
                                   start_word='小猫', show_process=False)
    print(f"   未训练输出: '{untrained_text}'")
    print("   （基本是随机的，没有语言逻辑）\n")

    # 4. 训练模型
    losses = train_model(model, training_pairs, epochs=100)

    # 5. 分析学习结果
    analyze_learned_patterns(model, vocab, id_to_word)
    demonstrate_meaning_space(model, vocab, id_to_word)

    # 6. 训练后生成测试
    print("🧠 训练后的智能生成:")

    test_starts = ['小猫', '鸟儿', '小狗']
    temperatures = [0.1, 0.8, 1.5]

    for start_word in test_starts:
        print(f"📝 以'{start_word}'开始的生成:")
        for temp in temperatures:
            result = generate_text(model, vocab, id_to_word,
                                   start_word=start_word,
                                   temperature=temp,
                                   show_process=False)
            print(f"   温度{temp}: '{result}'")
        print()

    # 7. 哲学总结
    philosophical_reflection(model, vocab)

    print("✨ 实验结论:")
    print("   通过这个极简模型，我们看到了智能的基本机制：")
    print("   - 符号到意义的映射（词嵌入）")
    print("   - 模式识别与预测（线性变换）")
    print("   - 创造性生成（概率采样）")
    print("   - 从经验到智慧（参数学习）")
    print()
    print("   正如老子所言：'道生一，一生二，二生三，三生万物'")
    print("   简单的数学原理，经过层层组合，最终通向了人工智能的无限可能。")

    return model, vocab, id_to_word


# 运行完整实验
if __name__ == "__main__":
    model, vocab, id_to_word = main()
