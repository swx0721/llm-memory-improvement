from typing import List
from .history_store import HistoryStore

def build_prompt(system_role: str,  query: str, evidence: str="") -> str:
    system = (
        f"系统指令：你是{system_role}。如果历史与当前指令冲突，以‘用户纠正’为准；"
        f"当证据不足时要明确说明，不要编造。回答时尽量引用[证据#编号]。"
    )
    context = f"历史证据:{evidence}" if evidence else "历史证据：未收集到历史证据"
    task = f"用户问题：{query}"

    prompt = (
        f"{system}\n{task}\n{context}\n请基于上述证据回答，并在关键处标注[证据#x]。"
    )
    return prompt

def get_evidence(history_store: HistoryStore, session_id: str, n_turns: int=5) -> str:
    recent_history = history_store.get_recent_history(session_id, n_turns=n_turns)
    return recent_history