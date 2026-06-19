"""下载 bert-base-chinese-medical-ner 模型到本地"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "iioSnail/bert-base-chinese-medical-ner"
local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bert-base-chinese-medical-ner")
os.makedirs(local_dir, exist_ok=True)

print(f"下载模型: {model_name}")
print(f"保存到: {local_dir}")
print(f"使用镜像: {os.environ['HF_ENDPOINT']}")

print("正在下载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)
print("Tokenizer 下载完成")

print("正在下载 Model...")
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.save_pretrained(local_dir)
print("Model 下载完成")

files = os.listdir(local_dir)
total_size = sum(os.path.getsize(os.path.join(local_dir, f)) for f in files) / 1024 / 1024
print(f"\n本地文件: {files}")
print(f"总大小: {total_size:.1f} MB")
print("\n模型下载成功!")
