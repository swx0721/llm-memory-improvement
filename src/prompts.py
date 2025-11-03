from typing import List


def build_prompt(system_role: str, evidence: str, query: str) -> str:
    system = (
        f"系统指令：你是{system_role}。如果历史与当前指令冲突，以‘用户纠正’为准；"
        f"当证据不足时要明确说明，不要编造。回答时尽量引用[证据#编号]。\n"
    )
    context = f"历史证据:\n{evidence}\n" if evidence else ""
    task = f"用户问题：{query}\n请基于上述证据回答，并在关键处标注[证据#x]。"
    return "\n".join([system, context, task])
