from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# =========================
# 関数定義
# =========================
def run_expert_chat(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプをもとにLLMに問い合わせ、回答を返す関数
    """
    # 専門家の種類ごとにシステムメッセージを定義
    system_messages = {
    "歴史の専門家": "あなたは歴史の専門家です。歴史的な出来事や背景を詳しく説明してください。",
    "栄養学の専門家": "あなたは栄養学の専門家です。食事や栄養に関する専門的なアドバイスをしてください。",
    "法律の専門家": "あなたは法律の専門家です。法的な観点から説明してください。",
    "教育の専門家": "あなたは教育の専門家です。学習指導や教育理論について説明してください。",
    "医療の専門家": "あなたは医療の専門家です。症状や健康についてわかりやすく説明してください。",
}


    system_template = system_messages.get(expert_type, "あなたは有能なアシスタントです。")

    # プロンプトテンプレート作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{user_input}")
    ])

    # LLMを定義
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # メモリを作成
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # チェーン作成部分を修正
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # 実行
    response = chain.run(user_input=user_input)

    return response


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="専門家チャットアプリ", layout="centered")

    # アプリの説明
    st.title("🧑‍🏫 専門家チャットアプリ")
    st.write("""
    このアプリでは、専門家の視点から質問に答えてくれるAIと対話できます。  
    以下の手順で利用してください：  
    1. 専門家の種類を選択  
    2. テキストを入力  
    3. 送信ボタンを押す  
    """)

    # 専門家の種類を選択
    expert_type = st.radio(
        "専門家の種類を選んでください：",
        ["歴史の専門家", "栄養学の専門家", "法律の専門家", "教育の専門家", "医療の専門家"]
    )

    # 入力フォーム
    examples = {
    "歴史の専門家": "例: 織田信長の外交戦略を教えて",
    "栄養学の専門家": "例: ビタミンCを多く含む食品を教えて",
    "法律の専門家": "例: 労働契約の更新に関するルールは？",
    "教育の専門家": "例: 効果的な英単語の覚え方は？",
    "医療の専門家": "例: 睡眠不足が体に与える影響を教えて",
    }

    placeholder_text = examples.get(expert_type, "質問を入力してください")
    
    user_input = st.text_area("質問を入力してください：", placeholder=placeholder_text)

    # 送信ボタン
    if st.button("送信"):
        if user_input.strip() == "":
            st.warning("質問を入力してください。")
        else:
            with st.spinner("AIが回答を生成中です..."):
                answer = run_expert_chat(user_input, expert_type)
            st.success("回答:")
            st.write(answer)


if __name__ == "__main__":
    main()