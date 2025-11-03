from src.embeddings import Embedder
from src.faiss_index import FaissIndex
from src.cache_store import CacheStore
from src.retriever import Retriever
import yaml

CFG_PATH = "config.yaml"


def test_retrieval_smoke():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    embedder = Embedder(cfg.get("embedding", {}).get("model"), device="cpu", batch_size=16)
    index = FaissIndex(cfg.get("faiss", {}).get("dim"), cfg.get("faiss", {}).get("index_path", "") + ".test")
    store = CacheStore(cfg.get("storage", {}).get("sqlite_path", "").replace(".sqlite", ".test.sqlite"))
    retriever = Retriever(embedder, index, store, topk=3)

    retriever.add_history("user", "我叫小王")
    retriever.add_history("assistant", "你好，小王")
    retriever.add_history("user", "明天去开会")

    res = retriever.search("我叫什么？")
    assert len(res) >= 1
    assert any("我叫小王" in r["text"] for r in res)
