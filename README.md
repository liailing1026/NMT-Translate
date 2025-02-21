# 神经机器翻译（NMT）中英文翻译
本项目实现了一个基于 **Transformer** 架构的神经机器翻译（NMT）模型，用于中文和英文之间的翻译。模型使用了 **SentencePiece** 分词器、**束搜索（Beam Search）** 解码器，使用 **PyTorch** 进行深度学习框架实现，包含了从数据预处理到模型训练、评估和翻译的完整流程。
## 技术栈
### 1. Transformer 架构
Transformer 是一种基于自注意力机制的神经网络架构，广泛用于自然语言处理（NLP）任务。它的核心思想是通过注意力机制来建模序列数据中的依赖关系，摒弃了传统循环神经网络（RNN）或卷积神经网络（CNN）的限制。

Transformer 模型由 **编码器（Encoder）** 和 **解码器（Decoder）** 两部分组成，每一部分都由多个 **层（Layer）** 堆叠而成。每一层包括：
- **多头自注意力机制（Multi-Head Attention）**：该机制使模型能够关注输入序列的不同部分，以捕捉长距离依赖关系。
- **前馈神经网络（Feed Forward Network）**：对每个位置的表示进行处理，提升模型的表达能力。

本项目实现的模型结构主要基于此 Transformer 架构，具体见 `model.py` 文件中的实现。

### 2. SentencePiece 分词器

SentencePiece 是一个用于无监督文本分词的工具，特别适用于处理字符级别的分词任务。与传统的分词方法（如基于词典的分词）不同，SentencePiece 使用子词（subword）单元来进行文本编码，有助于处理稀有词和未知词。

在本项目中，SentencePiece 被用来对英文和中文文本进行分词，模型训练过程中将使用这些分词单元来构建输入和输出。分词模型由 `tokenize.py` 脚本训练生成，训练过程中可以指定词汇表大小和分词模型类型。

### 3. 束搜索（Beam Search）

束搜索是一种启发式的搜索算法，用于在解码阶段寻找最优的输出序列。在机器翻译中，束搜索可以同时探索多个可能的翻译路径，避免贪婪解码可能错过的最佳结果。该方法通过设定束宽（beam size），在每一步选择得分最好的多个候选序列进行扩展，直到解码完成。

本项目在 `beam_decoder.py` 中实现了束搜索解码器。您可以在翻译过程中选择使用束搜索（beam search）或贪婪解码（greedy decoding）。

### 4. PyTorch 深度学习框架

PyTorch 是一个开源的深度学习框架，提供了强大的自动求导功能和灵活的神经网络构建能力。本项目使用 PyTorch 构建 Transformer 模型，并利用其 GPU 加速功能提高训练效率。

## 安装

安装该项目所需的依赖项：

```bash
pip install torch sentencepiece sacrebleu

项目结构.
├── config.py            # 配置文件，包含模型和训练参数
├── data_loader.py       # 数据加载和预处理函数
├── get_corpus.py        # 脚本，将原始 JSON 格式的语料转换为文本文件
├── main.py              # 主脚本，用于训练、评估和翻译
├── model.py             # 模型架构（Transformer）
├── tokenize.py          # 分词器训练与测试
├── train.py             # 训练和评估函数
├── utils.py             # 工具函数，日志记录和分词器加载
└── data/
    ├── corpus.ch        # 中文语料文件
    ├── corpus.en        # 英文语料文件
    ├── json/            # 存放训练、验证、测试 JSON 格式数据的文件夹
    └── tokenizer/       # 存放训练好的 SentencePiece 模型的文件夹
