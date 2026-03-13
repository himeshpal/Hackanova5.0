from qdrant_client import QdrantClient

# 1. Connect to your local database folder
client = QdrantClient(path="./local_qdrant_db")

# =========================================================
# STAGE 1: HARVESTED METADATA (Top 10)
# =========================================================
print("\n" + "="*70)
print(" 📚 TOP 10 FROM 'harvested_metadata' (STAGE 1)")
print("="*70)
try:
    harvested, _ = client.scroll(
        collection_name="harvested_metadata",
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    for i, record in enumerate(harvested, 1):
        print(f"{i}. ID: {record.payload.get('paper_id')}")
        print(f"   Title: {record.payload.get('title')[:100]}...")
        print("-" * 70)
except Exception as e:
    print(f"Collection 'harvested_metadata' not found or empty. {e}")

# =========================================================
# STAGE 2: EXTRACTED INTELLIGENCE (Top 10)
# =========================================================
print("\n" + "="*70)
print(" 🧠 TOP 10 FROM 'extracted_intelligence' (STAGE 2)")
print("="*70)
try:
    extracted, _ = client.scroll(
        collection_name="extracted_intelligence",
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    for i, record in enumerate(extracted, 1):
        print(f"{i}. ID: {record.payload.get('paper_id')}")
        print(f"   Methodology: {record.payload.get('methodology')[:100]}...")
        print(f"   Datasets: {record.payload.get('datasets')}")
        print("-" * 70)
except Exception as e:
    print(f"Collection 'extracted_intelligence' not found or empty. {e}")
print("\n")