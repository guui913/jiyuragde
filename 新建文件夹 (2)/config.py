import os
import logging

# === 项目根目录 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 配置日志
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                                      # 控制台输出
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),  # 文件持久化
    ],
)

# === 路径配置 ===
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
EVAL_RESULTS_DIR = os.path.join(DATA_DIR, "evaluation_results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FINETUNED_DIR = os.path.join(MODELS_DIR, "finetuned_lora")
MERGED_MODEL_DIR = os.path.join(MODELS_DIR, "merged_model")
CHROMA_PERSIST_DIR = os.path.join(PROCESSED_DIR, "chroma_db")
CHUNKS_JSON_PATH = os.path.join(PROCESSED_DIR, "chunks_metadata.json")

# === 知识库配置 ===
CHUNK_SIZE = 512         # 每个文本块的字符数
CHUNK_OVERLAP = 128      # 块之间的重叠字符数
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 嵌入模型
EMBEDDING_DIM = 512      # bge-small 输出维度
CHROMA_COLLECTION_NAME = "domain_knowledge_base"

# === 检索配置 ===
HYBRID_TOP_K_KEYWORD = 10  # 关键词检索返回Top-K
HYBRID_TOP_K_VECTOR = 10   # 向量检索返回Top-K
FINAL_TOP_K = 3            # 最终融合后Top-K
RRF_K = 60                 # RRF融合常数

# === 生成配置 ===
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05

# === 微调配置 ===
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024
USE_4BIT = True  # QLoRA量化微调

# === Ollama 配置 ===
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "llama3.2:latest"  # 默认模型，可改为 deepseek-r1:1.5b / qwen2.5:1.5b 等

# === API配置（备选方案） ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_API_MODEL = "deepseek-chat"

# === 系统提示词 ===
SYSTEM_PROMPT = """你是一个领域专业知识助手。你的回答必须：
1. 严格基于提供的参考资料
2. 如果资料不足以回答，请明确说"根据现有资料无法确定"
3. 回答专业、准确、简洁"""

# === 确保目录存在 ===
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, TRAINING_DIR,
          EVAL_RESULTS_DIR, MODELS_DIR, FINETUNED_DIR, MERGED_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
