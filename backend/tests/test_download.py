"""测试下载中文医疗NER模型"""
import os
import sys
from pathlib import Path

def test_download():
    print("=" * 60)
    print("测试下载中文医疗NER模型")
    print("=" * 60)

    os.environ["HF_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "0"

    try:
        print("\n1. 导入 transformers...")
        from transformers import pipeline

        print("\n2. 尝试下载模型 (设置30秒超时)...")
        os.environ["TRANSFORMERS_TIMEOUT"] = "30"

        print("\n正在下载模型 iioSnail/bert-base-chinese-medical-ner...")
        nlp = pipeline(
            "token-classification",
            model="iioSnail/bert-base-chinese-medical-ner",
            aggregation_strategy="simple"
        )

        print("\n✅ 模型下载成功!")
        print(f"   模型类型: {type(nlp).__name__}")

        print("\n3. 测试实体识别...")
        test_text = "患者因胸痛入院，既往有高血压病史10年"
        entities = nlp(test_text)
        print(f"   输入文本: {test_text}")
        print(f"   识别到 {len(entities)} 个实体:")
        for entity in entities:
            print(f"     - {entity['entity_group']}: {entity['word']}")

        print("\n✅ 测试完成!")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_download()
