from typing import List, Dict


def assemble_evidence(items: List[Dict], max_chars: int = 1200) -> str:
    # 简单策略：按 turn_id 升序、拼接并限长；带上编号，便于回答时引用
    items = sorted(items, key=lambda x: x["turn_id"])  # 时间顺
    blocks = []
    total = 0
    for i, it in enumerate(items, 1):
        block = f"[证据#{i} | turn={it['turn_id']} | {it['speaker']}]\n{it['text']}\n"
        if total + len(block) <= max_chars:
            blocks.append(block)
            total += len(block)
        else:
            break
    return "\n".join(blocks)
