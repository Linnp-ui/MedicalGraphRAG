"""使用中文医疗NER模型测试文档摄入流程"""
import sys
import os
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.medical_processor import MedicalTextProcessor
from src.ingestion.text_splitter import TextSplitter, SplitStrategy

def test_document_ingestion_pipeline():
    print("=" * 60)
    print("文档摄入流程测试 (使用中文医疗NER模型)")
    print("=" * 60)

    medical_file = Path(__file__).parent.parent / "data" / "input" / "medical_sample.txt"

    if not medical_file.exists():
        print(f"❌ 样本文件不存在: {medical_file}")
        return

    print(f"\n📄 加载文档: {medical_file.name}")

    loader = DocumentLoader()
    result = loader.load_safe(str(medical_file))

    if not result.success:
        print(f"❌ 文档加载失败: {result.error}")
        return

    document = result.document
    print(f"✅ 文档加载成功!")
    print(f"   标题: {document.title}")
    print(f"   内容长度: {len(document.content)} 字符")

    print(f"\n🔧 医疗文本预处理...")
    processor = MedicalTextProcessor()
    document = processor.process_document(document)
    print(f"✅ 预处理完成!")

    print(f"\n🏷️ 实体识别 (NER模型)...")
    entities = processor.extract_medical_entities(document.content)

    print(f"\n识别到的实体 ({len(entities)}个):")
    entity_types = {}
    for entity in entities:
        e_type = entity['type']
        entity_types[e_type] = entity_types.get(e_type, 0) + 1
        if e_type in ['DISEASE', 'SYMPTOM', 'DRUG', 'EXAMINATION']:
            print(f"  [{e_type:10}] {entity['text']}")

    print(f"\n实体类型统计:")
    for e_type, count in sorted(entity_types.items()):
        print(f"  {e_type}: {count}")

    print(f"\n📑 文本分割 (医疗策略)...")
    splitter = TextSplitter(strategy=SplitStrategy.MEDICAL)
    chunks = splitter.split_text(document.content, document.id)

    print(f"\n分割结果 ({len(chunks)}个块):")
    for i, chunk in enumerate(chunks[:5], 1):
        preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
        print(f"  块{i}: {preview}")

    if len(chunks) > 5:
        print(f"  ... 还有 {len(chunks) - 5} 个块")

    print(f"\n" + "=" * 60)
    print("✅ 文档摄入流程测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_document_ingestion_pipeline()
