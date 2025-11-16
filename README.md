# 📈 네이버 증권뉴스 RAG 챗봇  
네이버 증권 뉴스 크롤링 → 벡터 DB 임베딩 → RAG 분석 → 워드클라우드 → 자동 요약 → 질의응답  
Streamlit 기반 인터랙티브 대시보드

---

## 🧾 프로젝트 개요

이 프로젝트는 **네이버 증권 주요 뉴스**를 자동으로  
**크롤링 → 임베딩 → 벡터검색 → RAG 분석**하여 다음 기능을 제공합니다.

- 날짜별 뉴스 자동 크롤링  
- 기사 제목 기반 워드클라우드 생성  
- 전체 뉴스 50~100자 요약  
- 상위 키워드 기반 자동 분석 버튼  
- 직접 입력하는 질의응답 (RAG 기반)  
- Streamlit UI 기반 시각적 분석 대시보드

---

## 🚀 주요 기능

### 🔍 1. 네이버 증권 뉴스 크롤링
- 주요 뉴스 페이지 자동 순회  
- 제목 · 본문 수집  
- CSV로 자동 저장  

### 📄 2. LangChain RAG 파이프라인
- 뉴스 문서를 Document로 변환  
- Chunk 분할 (300자 + overlap 50)  
- FAISS 벡터스토어 구축  
- OpenAI Embeddings 생성  
- RAG 기반 질의응답 지원  

### 🔠 3. 자연어 처리 + WordCloud
- Kiwi(Kiwipiepy)로 명사 추출  
- 불용어 제거  
- WordCloud 시각화  

### 📰 4. 전체 뉴스 요약
- GPT로 50~100자 요약 자동 생성  

### 🧠 5. 상위 키워드 기반 자동 분석
- 워드클라우드 상위 10개 키워드 추출  
- 각 키워드에 대해 “50자 요약 분석” 버튼 자동 생성  

### 💬 6. 직접 입력 질의응답
- RAG 기반으로 뉴스 기반 답변 제공  

---


---

## 🛠 사용 기술 스택

| 분야 | 기술 |
|------|------|
| Web UI | Streamlit |
| Data Processing | pandas, re, requests, BeautifulSoup |
| NLP | Kiwipiepy, WordCloud |
| Embeddings | OpenAIEmbeddings |
| Vector DB | FAISS |
| LLM | ChatOpenAI (gpt-4o-mini) |
| Visualization | matplotlib |

---




