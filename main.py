from src.llm_client import ApiLLMClient
from src.prompts import build_prompt
from src.history_store import HistoryStore
import yaml

CFG_PATH = "config.yaml"


def bootstrap():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)
    llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key"),
        base_url=cfg.get("llm", {}).get("base_url"),
        model=cfg.get("llm", {}).get("model"),
    )

    # 初始化历史存储
    db_path = cfg.get("storage", {}).get(
        "history_db_path", "./cache/conversation_history.db"
    )
    history_store = HistoryStore(db_path=db_path)

    return cfg, llm, history_store


def chat_loop(cfg, llm, history_store):
    """连续对话循环，支持历史检索"""
    print("\n=== 开始对话 (输入 'quit' 或 'exit' 退出) ===\n")

    # 开始新的会话
    session_id = history_store.start_session()
    print(f"会话ID: {session_id}")
    turn_number = 0

    while True:
        # 获取用户输入
        user_input = input("用户: ").strip()

        if user_input.lower() in ["quit", "exit", "退出"]:
            print(f"对话结束，本次会话共进行了 {turn_number} 轮对话")
            print(f"历史记录已保存到数据库")
            break

        if not user_input:
            continue

        # 增加轮数
        turn_number += 1

        # 3) 构建 prompt
        prompt = build_prompt("耐心的助理", user_input)
        print(
            f"----------<构建的Prompt>----------\n{prompt}\n----------<构建的Prompt>----------"
        )

        # 4) 调用大模型生成回复
        try:
            answer = llm.generate(
                prompt, temperature=cfg.get("llm", {}).get("temperature")
            )
            print(f"助手: {answer}")

            # 5) 保存对话历史到数据库
            history_store.save_turn(session_id, turn_number, user_input, answer)
            print(f"[第{turn_number}轮对话已保存]")

        except Exception as e:
            print(f"错误: {str(e)}")
            turn_number -= 1  # 出错则不计入轮数
            continue
        print("----------<本轮对话结束>----------\n")


if __name__ == "__main__":
    cfg, llm, history_store = bootstrap()

    # 启动连续对话
    chat_loop(cfg, llm, history_store)
