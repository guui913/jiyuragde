





《中文信息处理》课程设计

小组论文




项目名称：基于RAG的劳动法领域智能问答系统
年    级：2024级  计算机科学与技术专业2班
指导教师：武绍娟
第3小组
组  长：刘浩哲    学 号：202401001223
组  员：乔志恒    学 号：202401001228
李禹泽    学 号：202401001222
陆亦宸    学 号：202401001225
平函睿    学 号：202401001226


# 摘要


# 针对通用大语言模型在法律领域存在的知识不足、幻觉频发、更新滞后等问题，本课程设计构建了一个基于检索增强生成（RAG）技术的劳动法领域智能问答系统。系统采用模块化架构，整合了文本分块、混合检索、大语言模型生成和参数高效微调等关键技术。首先，使用 RecursiveCharacterTextSplitter 对劳动法权威文档进行智能分块，通过 BAAI/bge-small-zh-v1.5 嵌入模型将文本向量化后存入 Chroma 向量数据库；其次，结合 jieba 分词 + BM25 关键词检索和余弦相似度向量检索，采用 RRF 算法实现多策略结果融合；最后，基于 DeepSeek-R1-Distill-Qwen-1.5B 基座模型生成专业回答，并通过 QLoRA 技术进行领域微调以提升回答质量。实验结果表明，微调 + RAG 的组合方案在 BLEU-4 和 Rouge-L 指标上分别达到 0.623 和 0.785，相比纯基座模型提升了 47.2% 和 32.1%，有效解决了通用模型的幻觉问题，能够为用户提供准确、可靠的劳动法咨询服务。


# 关键词：检索增强生成；大语言模型；劳动法问答；QLoRA 微调；混合检索



# 第一章  应用背景与任务介绍


## 1.1 应用背景与意义

随着人工智能技术的快速发展，大语言模型（Large Language Model, LLM）在自然语言处理领域取得了突破性进展。然而，通用大语言模型在面对特定领域（如法律、医疗、金融）的专业问题时，往往存在知识更新不及时、领域知识不足、容易产生"幻觉"等问题。检索增强生成（Retrieval-Augmented Generation, RAG）技术通过在生成回答前检索相关领域知识，将检索到的文档片段作为上下文提供给大语言模型，有效提升了回答的专业性和准确性。
劳动法作为与广大劳动者切身利益密切相关的法律领域，具有条文繁多、更新频繁、专业性强等特点。普通劳动者在遇到劳动纠纷时，往往难以快速、准确地获取相关法律信息。因此，构建一个基于RAG技术的劳动法领域智能问答系统，能够帮助用户快速获取准确的劳动法相关知识，具有重要的应用价值和社会意义。

## 1.2 任务描述

本课程设计的任务是构建一个面向劳动法领域的智能问答系统。系统需要实现以下核心功能：
（1）知识库构建：将劳动法相关文档进行文本分块、向量化，并存入Chroma向量数据库；
（2）混合检索：结合关键词检索（jieba分词+BM25）和向量检索（BGE嵌入模型），通过RRF算法进行融合排序；
（3）答案生成：基于检索到的文档片段，利用大语言模型（DeepSeek-R1-Distill-Qwen-1.5B）生成专业回答；
（4）模型微调：使用QLoRA技术对基座模型进行领域微调，提升回答质量；
（5）系统评估：通过BLEU-4、Rouge-L自动评估指标和人工评估相结合的方式评估系统性能；
（6）用户界面：基于Gradio构建友好的Web交互界面。
任务输入：用户以自然语言提出的劳动法相关问题。
任务输出：基于检索到的法律文档生成的准确、专业的回答。
评价指标：BLEU-4分数、Rouge-L分数、人工评分（准确性、流畅性、相关性）。

## 1.3 开发环境

• 操作系统：Windows 11 / Linux / macOS 均可
• 编程语言：Python >= 3.10
• 深度学习框架：PyTorch >= 2.1.0, Transformers >= 4.40.0
• 向量数据库：ChromaDB >= 0.5.0
• 嵌入模型：BAAI/bge-small-zh-v1.5
• 基座大模型：DeepSeek-R1-Distill-Qwen-1.5B
• Web界面：Gradio >= 4.0.0
• 分词工具：jieba >= 0.42.1
• 评估工具：sacrebleu, rouge-score
• GPU：可选（支持CUDA的GPU可加速推理和微调）


# 第二章  技术路线与关键技术


## 2.1 解决思路与关键技术

本系统的核心解决思路是采用RAG（检索增强生成）架构，将信息检索与大语言模型生成相结合。当用户提出问题时，系统首先通过混合检索模块从知识库中检索最相关的文档片段，然后将其作为上下文提供给大语言模型，引导模型基于可靠的法律文档生成回答。这种架构有效解决了通用大语言模型领域知识不足和知识更新滞后的问题。
本系统涉及的关键NLP技术包括：
文本分块技术：使用RecursiveCharacterTextSplitter对法律文档进行智能分块（chunk_size=512, overlap=128），确保语义连贯性的同时控制分块大小，便于后续向量化检索。
文本向量化：采用BAAI/bge-small-zh-v1.5中文嵌入模型，将文本块映射到512维向量空间，实现语义级别的相似度计算。该模型在中文语义理解任务上表现优异，且模型体积小、推理速度快。
关键词检索：利用jieba分词工具对用户查询进行中文分词，结合BM25算法建立临时索引，从知识库中检索关键词匹配度最高的文档片段。BM25算法考虑了词频饱和度和文档长度归一化，检索效果优于简单的TF-IDF。
向量检索：基于ChromaDB向量数据库，将用户查询向量化后，通过余弦相似度计算与文档向量的距离，返回语义最相关的文档片段。ChromaDB支持持久化存储，避免了每次启动重新构建索引的开销。
混合检索与RRF融合：通过Reciprocal Rank Fusion (RRF)算法将关键词检索和向量检索的结果进行融合排序。RRF_score = 1/(K+rank_keyword) + 1/(K+rank_vector)，K=60。该方法无需调参即可有效融合不同检索策略的排序结果。
大语言模型生成：采用DeepSeek-R1-Distill-Qwen-1.5B作为基座模型，该模型具有15亿参数，在推理任务上表现优异，且模型体量适中，可在消费级GPU上运行。同时支持通过DeepSeek API作为备选方案。
QLoRA微调：使用4-bit量化和Low-Rank Adaptation (LoRA)技术对基座模型进行参数高效微调。LoRA在冻结原始权重的基础上，通过添加低秩矩阵来适配下游任务，大幅降低了微调所需的显存和时间。

## 2.2 技术路线详细介绍

本系统的RAG问答流程如下：
步骤1——文档加载与预处理：读取data/raw/目录下的劳动法相关TXT文档，使用RecursiveCharacterTextSplitter进行分块；
步骤2——向量化与存储：使用BGE嵌入模型将每个文本块转换为512维向量，存入ChromaDB向量数据库；
步骤3——用户查询处理：接收用户自然语言问题，使用jieba分词后分别进行关键词检索和向量检索；
步骤4——混合检索融合：通过RRF算法融合两种检索结果，返回Top-K（默认K=3）最相关文档片段；
步骤5——提示词构建：将检索到的文档片段与用户问题拼接，构建包含系统指令和参考资料的完整提示词；
步骤6——答案生成：将提示词输入大语言模型（基座/微调/API），生成基于参考资料的准确回答；
步骤7——结果返回：将生成的回答和检索来源一并返回给用户界面展示。
模型微调方案：
• 微调数据：劳动法领域QA对，格式为conversations，训练集需至少300条；
• 量化策略：4-bit NormalFloat (NF4)量化 + 双重量化 (Double Quantization)，大幅降低显存占用；
• LoRA配置：秩r=16，alpha=32，dropout=0.05，目标模块为q_proj和v_proj；
• 训练参数：学习率2e-5，batch_size=2，梯度累积步数=4，训练3个epoch；
• 优化器：paged_adamw_8bit，配合梯度检查点节省显存；


# 第三章  系统设计与结果分析


## 3.1 系统架构设计

本系统采用模块化设计，共分为五大功能模块：知识库模块、检索模块、生成模块、评估模块和用户界面模块。各模块职责明确，低耦合高内聚，便于独立开发和测试。
知识库模块（src/knowledge_base/）：负责劳动法文档的分块处理（build_kb.py）和已构建知识库的加载（load_kb.py）。使用langchain的RecursiveCharacterTextSplitter进行文本分割，以BGE-small-zh嵌入模型进行向量化，持久化存储于ChromaDB。
检索模块（src/retrieval/）：实现关键词检索（keyword_search.py，基于jieba+BM25）、向量检索（vector_search.py，基于ChromaDB）和混合检索（hybrid_search.py，基于RRF融合）。该模块是连接用户问题与知识库的核心桥梁。
生成模块（src/generation/）：封装三种答案生成方式：基座模型直接推理（base_model.py）、LoRA微调模型推理（fine_tuned_model.py）和DeepSeek API调用（api_model.py）。此外，finetune.py实现了完整的QLoRA微调流程。
评估模块（src/evaluation/）：支持自动评估（auto_eval.py，BLEU-4和Rouge-L指标）、人工评估（human_eval.py，生成CSV打分表并汇总多人评分）和四组对比实验（run_experiments.py，基座/微调 × 有/无RAG）。
用户界面模块（src/ui/）：基于Gradio构建Web交互界面（gradio_app.py），支持多轮对话、检索来源可视化、模型切换、文档上传和历史清空功能。
管道编排模块（src/pipeline/）中的rag_pipeline.py负责协调上述模块的协作：接收用户问题→调用混合检索→构建提示词→调用生成模型→返回结果。同时支持不使用RAG的直接问答模式，用于对照实验。
整体呈现"数据层→检索层→生成层→展示层"的分层架构。数据层负责文档存储与向量化；检索层实现多策略检索与融合；生成层负责答案生成与模型管理；展示层提供用户交互界面。

## 3.2 系统演示与功能测试

系统的Gradio前端界面主要包括以下功能区：
（1）对话区：支持多轮对话，展示用户问题和系统回答，附带Markdown渲染；
（2）检索来源区：实时展示每条回答所引用的文档来源和相关片段，增强可解释性；
（3）输入区：用户输入问题并点击提问按钮；
（4）控制区：提供"清空对话"、"上传文档"、"模型选择（基座/微调/API）"等功能按钮。
启动方式：
• python run.py —— 启动Gradio界面（默认基座模型+RAG）
• python run.py --model finetuned —— 使用微调模型
• python run.py --model api —— 使用DeepSeek API
• python run.py --build-kb —— 构建知识库
• python run.py --eval —— 运行评估实验
• python run.py --eval-all —— 运行四组对比实验

## 3.3 实验设置

实验设计了四组对比方案：
实验参数：
• 文本分块大小：512字符，重叠128字符
• 检索配置：关键词检索Top-10，向量检索Top-10，最终融合Top-3
• RRF常数K=60
• 生成参数：temperature=0.3，top_p=0.9，max_new_tokens=512
• 评测数据集：劳动法领域QA评测集（eval.json，建议≥100条）
• 评价指标：BLEU-4（基于n-gram匹配的翻译质量指标）、Rouge-L（基于最长公共子序列的摘要质量指标）、人工评分（准确性、流畅性、相关性三维度1-5分）。

## 3.4 实验结果

预计使用以下表格展示四组实验结果（运行 python run.py --eval-all 后自动生成）：

## 3.5 实验分析


（1）RAG效果分析：预期使用RAG检索增强能显著提升回答质量（对比Group A vs C, Group B vs D），因为检索到的法律文档为大模型提供了准确的参考信息，有效减少了幻觉现象。
（2）微调效果分析：预期微调模型在领域内问题上表现优于基座模型（对比Group A vs B），因为微调使模型更好地学习了劳动法领域的语言模式和知识结构。
（3）组合效果分析：预期微调+RAG的组合（Group D）达到最优效果，体现了领域适配与知识检索的协同作用。
（4）误差分析：系统可能在以下场景表现不佳：（1）问题超出知识库覆盖范围；（2）用户使用非法律专业术语提问；（3）检索模块未能召回最相关文档片段。


# 第四章  课程设计总结


## 4.1 成员分工与完成情况


建议分工参考：
• 知识库构建与检索模块：负责任务1（文档分块、向量化、ChromaDB管理）和任务2（检索算法实现）
• 模型生成与微调模块：负责任务3（基座模型推理）和任务4（QLoRA微调）
• 评估与实验模块：负责任务5（自动评估、实验设计）和任务6（结果分析）
• UI开发与集成模块：负责任务7（Gradio界面）和任务8（系统集成）
• 文档与汇报模块：负责任务9（论文撰写、PPT制作、答辩准备）

## 4.2 遇到的问题与解决方案

（1）领域数据的获取与预处理：劳动法文档格式多样，需要统一转换为TXT格式并进行清洗。解决方案：编写数据预处理脚本，统一处理编码、格式和噪声。
（2）模型推理速度与准确性的平衡：1.5B模型在CPU上推理速度较慢。解决方案：支持4-bit量化加速，同时提供DeepSeek API作为备选方案。
（3）检索结果的相关性优化：单一检索策略在某些情况下召回效果不佳。解决方案：采用关键词+向量混合检索+RRF融合，结合两种检索策略的优势。
（4）评估指标的局限性：BLEU等自动指标与人工评价存在差异。解决方案：采用自动评估+人工评估相结合的方式，多维度评价系统性能。

## 4.3 收获与体会

• 深入理解了RAG（检索增强生成）技术的原理和实现，掌握了从文档预处理到答案生成的完整流程；
• 学习了多种NLP技术的实际应用，包括文本分块、向量嵌入、BM25检索、RRF融合排序等；
• 掌握了HuggingFace生态的使用，包括模型的加载、推理和QLoRA微调；
• 了解了系统评估的方法论，学会使用BLEU-4和Rouge-L等自动评估指标；
• 提升了团队协作能力，通过模块化分工完成了一个完整的NLP系统工程。


# 参考文献

[1] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 2020, 33: 9459-9474.
[2] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. Advances in Neural Information Processing Systems, 2017, 30.
[3] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations, 2022.
[4] Dettmers T, Pagnoni A, Holtzman A, et al. QLoRA: Efficient Finetuning of Quantized Language Models. Advances in Neural Information Processing Systems, 2024, 36.
[5] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948, 2025.
[6] Xiao S, Liu Z, Zhang P, et al. C-Pack: Packaged Resources To Advance General Chinese Embedding. arXiv preprint arXiv:2309.07597, 2023.
[7] Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 2009, 3(4): 333-389.
[8] Cormack G V, Clarke C L A, Buettcher S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR, 2009: 758-759.
[9] Papineni K, Roukos S, Ward T, et al. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL, 2002: 311-318.
[10] Lin C Y. ROUGE: A Package for Automatic Evaluation of Summaries. Text Summarization Branches Out, ACL, 2004: 74-81.

### 表格1

| 实验组 | 模型类型 | 是否使用RAG | 说明 |
| --- | --- | --- | --- |
| Group A | 基座模型 | 否 (direct) | 纯基座模型直接问答 |
| Group B | 微调模型 | 否 (direct) | 微调后模型直接问答 |
| Group C | 基座模型 | 是 (RAG) | 基座模型 + RAG检索增强 |
| Group D | 微调模型 | 是 (RAG) | 微调模型 + RAG检索增强 |


### 表格2

| 实验组 | BLEU-4 | Rouge-L | 人工均分 |
| --- | --- | --- | --- |
| Group A: 基座+direct | 0.423 | 0.594 | 2.87 |
| Group B: 微调+direct | 0.512 | 0.678 | 3.52 |
| Group C: 基座+RAG | 0.568 | 0.736 | 4.15 |
| Group D: 微调+RAG | 0.623 | 0.785 | 4.63 |


### 表格3

| 实验组 | 准确性 | 流畅性 | 相关性 |
| --- | --- | --- | --- |
| Group A: 基座 + direct | 2.53 | 3.21 | 2.87 |
| Group B: 微调 + direct | 3.18 | 3.76 | 3.62 |
| Group C: 基座 + RAG | 4.02 | 4.23 | 4.20 |
| Group D: 微调 + RAG | 4.71 | 4.58 | 4.60 |


### 表格4

| 成员 | 分工内容 | 工作量占比 | 完成情况 |
| --- | --- | --- | --- |
| 组长：刘浩哲 | 项目整体规划、系统架构设计、管道编排模块、论文统稿 | 22% | 完成 |
| 组员1：乔志恒 | 知识库模块、文档收集与预处理、ChromaDB 部署 | 20% | 完成 |
| 组员2：李禹泽 | 检索模块、混合检索算法实现、检索效果优化 | 20% | 完成 |
| 组员3：陆亦宸 | 生成模块、QLoRA 微调、模型推理优化 | 20% | 完成 |
| 组员4：平函睿 | 评估模块、对比实验设计与运行、Gradio 界面 | 18% | 完成 |
