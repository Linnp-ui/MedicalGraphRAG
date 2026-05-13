"""下载并保存中文医疗NER模型到本地指定目录"""
import os
import shutil
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def download_and_save_model():
    model_name = "iioSnail/bert-base-chinese-medical-ner"
    local_path = Path(__file__).parent.parent / "models" / "bert-base-chinese-medical-ner"

    print("=" * 60)
    print("下载并保存中文医疗NER模型到本地")
    print("=" * 60)
    print(f"\n模型名称: {model_name}")
    print(f"保存路径: {local_path}")

    os.environ["HF_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "0"

    if local_path.exists():
        print(f"\n⚠️ 本地路径已存在，先删除...")
        shutil.rmtree(local_path)

    local_path.mkdir(parents=True, exist_ok=True)

    print("\n正在下载模型，这可能需要几分钟...")

    try:
        print("\n1. 下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_path)
        print("   ✅ tokenizer 下载完成")

        print("\n2. 下载模型权重...")
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model.save_pretrained(local_path)
        print("   ✅ 模型权重下载完成")

        print("\n3. 保存模型配置...")
        with open(local_path / "config.json", "w") as f:
            import json
            config = {
                "model_name": model_name,
                "architecture": "bert-base-chinese-medical-ner",
                "task": "token-classification",
                "aggregation_strategy": "simple"
            }
            json.dump(config, f, indent=2)
        print("   ✅ 配置文件保存完成")

        print(f"\n✅ 模型保存成功!")
        print(f"\n📁 模型文件列表:")
        for file in local_path.iterdir():
            size = file.stat().st_size / (1024 * 1024)
            print(f"   {file.name}: {size:.2f} MB")

        print(f"\n代码中使用:")
        print(f"   from transformers import pipeline")
        print(f"   pipe = pipeline('token-classification', model='{local_path}')")

    except Exception as e:
        print(f"\n❌ 保存失败: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    download_and_save_model()
