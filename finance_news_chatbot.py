from datetime import datetime
import time
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


# 환경변수 로드
load_dotenv(".env")

# 폴더 설정
DATA_DIR = "./naver_finance_news"
VECTORSTORE_DIR = "faiss_index"


# ================================================
# 1. CSV 파일 → Document 변환 + Chunk 나누기
# ================================================
def load_csv_and_split(date_str):
    csv_path = os.path.join(DATA_DIR, f"{date_str}.csv")

    if not os.path.exists(csv_path):
        return None, f"CSV 파일이 없습니다: {csv_path}"

    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        subject = str(row["subject"])
        content = str(row["content"])

        # 제목 + 본문으로 Document 구성
        text = f"[제목] {subject}\n\n{content}"

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "subject": subject,
                    "content_length": len(content)
                }
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    return splitter.split_documents(docs), None



# ================================================
# 2. 벡터스토어 생성 (Batch 적용)
# ================================================
def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    batch_size = 100

    vectordb = None

    progress = st.progress(0)
    status = st.empty()

    total = (len(docs) + batch_size - 1) // batch_size

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        now_batch = i // batch_size + 1

        progress.progress(now_batch / total)
        status.text(f"{now_batch}/{total} 배치 임베딩 중...")

        if vectordb is None:
            vectordb = FAISS.from_documents(batch, embeddings)
        else:
            temp_db = FAISS.from_documents(batch, embeddings)
            vectordb.merge_from(temp_db)

    vectordb.save_local(VECTORSTORE_DIR)

    progress.empty()
    status.empty()

    return vectordb


# ================================================
# 3. 기존 벡터스토어 로드
# ================================================
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


# ================================================
# 4. RAG 체인 생성
# ================================================
def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    아래 참고 문서를 기반으로 사용자의 질문에 답변해 주세요.

    질문: {question}

    참고 문서:
    {context}
    """)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever,
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
    )
    return chain

# ================================================
# 5. 뉴스 100자 요약
# ================================================
def summarize_news(df):
    """전체 뉴스를 100자 내외로 요약"""
    all_text = " ".join(df["content"].astype(str).tolist())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    summary_prompt = f"""
    아래는 네이버 증권뉴스 전체 내용입니다.
    핵심 내용을 100자 내외로 간결하게 요약해 주세요.

    내용:
    {all_text[:10000]}  # 너무 길면 자르기 (토큰 보호)
    """

    response = llm.invoke([{"role":"user","content": summary_prompt}])
    return response.content


# ================================================
# 6. wordcloud 생성 
# ================================================
from kiwipiepy import Kiwi

kiwi = Kiwi()


KOREAN_STOPWORDS = set("""
가 각 간 감 값 것 겐 경우 게 결과 고 곳 과 관계 관련 관심 관해 거의 
그래 그러나 그러나 그래도 그래서 그리고 그러면 그런 그런지 그럼 그때 그때문에 
그런데 그럼에도 불구하고 글고 기타 그냥 그나마 그럼에 따라 그뿐이다 그밖에 그중 그동안 
나 남 너무 년 년대 내내 넷 누구 다 다시 단 단지 대 대다 대부분 더욱 더욱더 
더더욱 때문에 또 또는 또한 때 때로 때마다 등 등이 따라 또는 따름만 따름이다 
라 로 로써 를 모든 및 바 바람 반 여러 여러가지 여러개 여러번 여러해 어 어도 
엄청 여 여전히 역시 오 오히려 와 왜 외 외에 요 우리 우리나라 우리들 우리가 위 위하여 
위해 위한 으로 은 이는 이번 이래 이러 이러이러한 이러한 이런 이라고 이러한데 
이름 이후 이외 이와 이런저런 이젠 이제 일 일단 일반 일반적 일반적으로 이미 이외 
자 전 전혀 전체 전체적 전체적으로 제 주변 지금 즉 지만 진짜 제대로 줄 중 중에 
지 금 가장 가장으로 가장은 가장은 잘 잘못 잘못된 적 적어도 적절 절대 절대로 주로 집근처 
처 처음 첫 첫째 최근 참고 통해 통해 틀림없이 편 포 이상 이래서 이후로 이런들 
때문 때문이다 따라서 하지만 혹은 혹시 혹 있다 없다 없다면 많이 많은 매우 매우도 맨 
해야 한 한다 하는 하는데 하나 하나씩 한번 한번도 하여 하여금 하지만 하고 하며 하면 
하는데 하듯 하든지 하게 하도록 하자 하자마자 하진 혹시 혹은 혹시나 혹은 
""".split())
MODERN_COLORS = [
    "#0D1B2A",  # deep navy
    "#1B263B",  # navy
    "#415A77",  # blue-gray
    "#778DA9",  # soft gray-blue
    "#E0E1DD"   # washed white
]

def extract_nouns(text: str):
    """Kiwi로 문장에서 명사만 추출"""
    nouns = []
    for token in kiwi.tokenize(text):
        if token.tag in ["NNG", "NNP"]:   # 일반명사/고유명사
            nouns.append(token.form)
    return nouns

def modern_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return MODERN_COLORS[hash(word) % len(MODERN_COLORS)]

def generate_wordcloud(df):
    # 1) 제목(subject)만 합치기
    text = " ".join(df["subject"].astype(str).tolist())

    # 2) 특수문자 제거
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3) Kiwi로 명사만 추출
    nouns = extract_nouns(text)

    # 4) 불용어 제거
    words = [
        noun for noun in nouns
        if noun not in KOREAN_STOPWORDS and len(noun) > 1
    ]

    cleaned_text = " ".join(words)

    font_path = "C:/Windows/Fonts/malgun.ttf"

    # 🔥 기본 디자인 워드클라우드
    wc = WordCloud(
        font_path=font_path,
        background_color="white",   # 기본 배경
        width=900,
        height=500,
        max_words=200,
    ).generate(cleaned_text)

    return wc


# ================================================
# 7. 크롤링
# ================================================
def crawl_naver_finance_news(date_str):
    page = 1
    article_list = []
    # 🔥 Streamlit 상태 표시: 여기에 한 줄만 계속 업데이트됨
    status = st.empty()
    while True:
        print(f"===== {page} 페이지 크롤링 중 =====")
        status.write(f"📡 {page} 페이지 크롤링 중...")

        url = f"https://finance.naver.com/news/mainnews.naver?date={date_str}&page={page}"

        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select(".block1")

        if not articles:
            break  # 더 이상 뉴스 없으면 종료

        for article in articles:
            subject = article.select_one(".articleSubject").text.strip()
            print(subject)
            status.write(f"📰 기사 수집: {subject}")
            # 링크 수집
            link = article.select_one(".articleSubject>a").attrs["href"]
            parsed = urlparse(link)
            params = parse_qs(parsed.query)
            article_id = params['article_id'][0]
            office_id = params['office_id'][0]
            news_link = f'https://n.news.naver.com/mnews/article/{office_id}/{article_id}'

            # 내용 수집
            detail_html = requests.get(news_link).text
            detail_soup = BeautifulSoup(detail_html, "html.parser")
            content = detail_soup.select_one("#dic_area").text.strip()

            time.sleep(0.5)

            article_list.append({
                "subject": subject,
                "content": content
            })
        
        if soup.select_one(".pgRR") is None:
            break

        page += 1
        time.sleep(1)

    # 저장
    df = pd.DataFrame(article_list)
    csv_path = f"./naver_finance_news/{date_str}.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, len(df)




# ================================================
# 8. Streamlit UI
# ================================================
# 세션 상태 초기화
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# ----------------------------------------
# 날짜 선택 UI
# ----------------------------------------
st.set_page_config(page_title="네이버 증권뉴스 RAG 챗봇", layout="wide")


# --- 사이드바 설정 ---
with st.sidebar:
    st.title("📈 네이버 증권뉴스 RAG 챗봇")
    st.header("⚙️ 설정")

    selected_date = st.date_input("날짜를 선택하세요", value=datetime.today())
    date_str = selected_date.strftime("%Y-%m-%d")

    if st.button("금일 뉴스 분석 실행 🚀"):
        csv_path = f"./naver_finance_news/{date_str}.csv"

        # 1) CSV 없으면 크롤링
        if not os.path.exists(csv_path):
            with st.spinner("CSV 파일이 없어 크롤링 중입니다..."):
                csv_path, count = crawl_naver_finance_news(date_str)
            st.success(f"{count}개의 기사를 크롤링해 CSV로 저장했습니다!")

        # 2) CSV 로드
        df = pd.read_csv(csv_path)
        st.session_state["df"] = df

        # 3) VectorDB 생성
        split_docs, error = load_csv_and_split(date_str)
        if error:
            st.error(error)
            st.stop()

        with st.spinner("벡터스토어 생성 중..."):
            vectordb = create_vectorstore(split_docs)
            st.session_state.vectordb = vectordb
            st.session_state.rag_chain = build_rag_chain(vectordb)

        # 4) 뉴스 요약 (50자로 바꿀 예정 – 2단계에서 수정)
        with st.spinner("뉴스 요약 생성 중..."):
            summary = summarize_news(df)
            st.session_state["summary"] = summary

        # 5) 워드클라우드 생성
        with st.spinner("워드클라우드 생성 중..."):
            wc = generate_wordcloud(df)
            st.session_state["wordcloud"] = wc

            # 🔥 상위 10개 키워드도 저장 (4단계에서 활용)
            top_keywords = sorted(wc.words_.items(), key=lambda x: x[1], reverse=True)[:10]
            st.session_state["top_keywords"] = [w for w, _ in top_keywords]

        st.success("분석 완료! 결과를 확인하세요 👇")

# 날짜/요일 표시
weekday_kor = ["월", "화", "수", "목", "금", "토", "일"][selected_date.weekday()]
display_date = selected_date.strftime("%Y-%m-%d")

st.markdown(
    f"""
    <h1 style="margin-top:0.5rem; margin-bottom:1rem;">
        📅 {display_date} ({weekday_kor}요일)
    </h1>
    """,
    unsafe_allow_html=True
)


if "summary" in st.session_state or "wordcloud" in st.session_state:
    st.markdown("---")
    st.subheader("📊 오늘의 시장 한눈에 보기")

    # 왼쪽: 워드클라우드, 오른쪽: 요약
    col1, col2 = st.columns([1, 1])  # 왼쪽 넓게 / 오른쪽 좁게

    # 왼쪽: 워드클라우드 카드
    with col1:
        if "wordcloud" in st.session_state:
            wc = st.session_state["wordcloud"]
            fig, ax = plt.subplots(figsize=(6, 4))  # 🔥 왼쪽이 넓으니까 조금 더 크게
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)

    # 오른쪽: 50자 요약 카드
    with col2:
        if "summary" in st.session_state:
            st.markdown(
                f"""
                <div style="
                    padding:1rem 1.25rem;
                    border-radius:1rem;
                    background-color:#F9FAFB;
                    border:1px solid #E5E7EB;
                    box-shadow:0 1px 3px rgba(15,23,42,0.08);
                    font-size:0.95rem;
                    line-height:1.5;
                    height:100%;
                ">
                    <div style="font-size:0.85rem; color:#6B7280; margin-bottom:0.25rem;">
                        📌 오늘 증권뉴스 50자 요약
                    </div>
                    <div style="font-weight:500; color:#111827;">
                        {st.session_state["summary"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ----------------------------------------
# 🔥 TOP 10 키워드 기반 자동 질문 생성
# ----------------------------------------
if "top_keywords" in st.session_state:

    st.markdown("### 🔍 주요 키워드 기반 자동 분석")

    keywords = st.session_state["top_keywords"][:10]

    btn_refs = []

    # 10개 키워드를 5개씩 나눔
    chunks = [keywords[i:i+5] for i in range(0, len(keywords), 5)]

    for chunk in chunks:
        cols = st.columns(len(chunk))
        for idx, kw in enumerate(chunk):
            with cols[idx]:
                if st.button(f"【{kw}】 이슈 50자 요약", key=f"kwbtn_{kw}"):
                    btn_refs.append(kw)

    # 버튼 눌렀을 때 자동 질문 실행
    if btn_refs:
        auto_question = f"{btn_refs[0]} 관련 이슈를 50자 내외로 요약해줘."
        st.write(f"**질문 자동 생성:** {auto_question}")
        with st.spinner("답변 생성 중..."):
            result = st.session_state.rag_chain.invoke({"question": auto_question})
        st.write(result.content)

    st.markdown("---")


# ----------------------------------------
# 🔥 직접 입력하는 질문 UI
# ----------------------------------------
st.subheader("🔍 뉴스 기반 질의응답 (직접 질문)")

if st.session_state.rag_chain:
    user_question = st.text_input("원하는 질문을 입력하세요")

    if user_question:
        with st.spinner("답변 생성 중..."):
            result = st.session_state.rag_chain.invoke({"question": user_question})

        st.write("### 💬 답변")
        st.write(result.content)
else:
    st.info("먼저 날짜를 선택하고 뉴스 분석을 실행하세요.")

a