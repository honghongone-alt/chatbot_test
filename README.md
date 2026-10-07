# RAG 챗봇 개발 환경

Python 3.11과 uv를 사용합니다. 의존성의 실제 설치 버전은 `uv.lock`에 고정되어 있습니다.

한글 Windows 경로에서 Python 3.11의 editable 설치 경로 디코딩 오류를 피하려면 uv 명령에 `--no-editable`을 사용하세요. `src/chatbot`의 코드를 변경하면 `uv sync --no-editable --reinstall-package chatbot`으로 다시 설치하세요. 프로젝트 루트의 `app.py`는 바로 실행할 수 있습니다.

## 환경 설치 및 확인

```powershell
uv sync --locked --no-editable
uv run --locked --no-editable python scripts/check_environment.py
```

확인 스크립트는 5개 패키지의 import, 텍스트 분할, dotenv 파싱을 검사하며 Python 경고도 오류로 처리합니다. API 호출은 하지 않습니다.

## API 키 설정

`.env.example`을 `.env`로 복사하고 `OPENAI_API_KEY`를 입력하세요. `.env`는 Git에서 제외됩니다.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
```

OpenAI 모델 객체는 API 키를 설정한 후 생성하세요.

## Streamlit 실행

프로젝트 최상단의 `run.bat`을 더블클릭하면 챗봇이 실행되고 브라우저가 열립니다. 이미 실행 중이면 기존 화면을 엽니다. 실행 창은 사용 중에 열어 두고, 종료하려면 창에서 `Ctrl+C`를 누르세요. uv가 설치되어 있어야 합니다.

다음 명령으로 실행한 뒤 브라우저에서 표시된 주소를 여세요.

```powershell
uv run --locked --no-editable streamlit run app.py
```

왼쪽의 **문서 읽기 / 다시 읽기**를 누르면 DATA 하위 폴더까지 전체 파일을 읽습니다. 현재 PDF 2개를 포함해 PDF, TXT, MD, CSV, JSON, LOG 파일을 지원합니다. 지원하지 않는 파일이나 텍스트 추출 실패는 화면에 알리며, 이미지 기반 PDF는 별도 OCR이 필요합니다.

`text-embedding-3-small` 임베딩과 `InMemoryVectorStore`로 검색하고 `gpt-4o-mini`로 답변합니다. Q&A 질문·답변·예외를 묶어 보존하며 긴 항목과 규정은 문단 경계를 우선해 최대 1,800자, 300자 중복으로 나눕니다. 파일명, 시작/끝 페이지, Q&A 번호, 제목을 저장합니다. 한국어 키워드 BM25와 벡터 검색 순위를 합쳐 후보를 찾고, 질문의 조건에 맞는 근거를 모델이 다시 선택합니다. 답변 생성 후 조건·금액·상한·예외와 인용의 의미를 검토하며 불일치하면 한 번 재생성하고 검증이 계속 실패하면 답변을 보류합니다. LangChain의 `invoke`, LCEL, 구조화된 출력을 사용합니다. 자동 검토도 모델을 사용하므로 정확성을 완벽하게 보장하지는 않습니다.

벡터DB는 브라우저 세션 메모리에 저장됩니다. 세션 또는 서버가 재시작되면 다시 읽어야 하며 OpenAI API 비용이 발생합니다. 파일이나 API 키를 바꾸면 검색 인덱스만 초기화합니다. 각 질문은 독립적으로 검색하므로 대상과 조건을 포함해 질문하세요.

대화와 출처는 `.chatbot/history.sqlite3`에 자동 저장되어 새로고침, 재실행, 문서 다시 읽기 후에도 유지됩니다. 왼쪽 **대화 초기화** 버튼을 누르면 저장된 대화만 지우고 검색 인덱스는 유지합니다. 이 로컬 앱을 여러 탭에서 열면 같은 대화 기록을 사용하며 새로고침 시 반영됩니다. 저장 폴더는 Git에서 제외됩니다. 과거 답변의 출처는 당시 문서 기준입니다.

## 앱 검증

`validate.bat`을 더블클릭하면 문서 추출, 검색, 기존 10개 질문과 조건을 바꾼 4개 질문의 실제 답변을 검증하고 HTML 보고서를 엽니다. 실제 OpenAI API 비용이 발생합니다. 사용자 대화 기록은 변경하지 않습니다. 하나라도 검사에 실패하면 종료 코드 1을 반환합니다.

터미널에서 `validate.bat offline`을 실행하면 API 없이 추출만 검증합니다. 오프라인 보고서는 `.chatbot/extraction_latest.html`과 `.chatbot/extraction_latest.json`에 별도로 저장됩니다. API 크레딧 부족은 통과로 처리하지 않고 보고서에 중단 사유를 기록합니다.

```powershell
# API 없이 페이지·Q&A 구조 검사
uv run --locked --no-editable python -X utf8 -W error scripts/validate_rag.py --offline

# 임베딩·검색·답변을 실제 API로 검증 (14개 질문)
uv run --locked --no-editable python -X utf8 -W error -u scripts/validate_rag.py --live
```

보고서는 `.chatbot/validation_latest.html`과 `.chatbot/validation_latest.json`에 저장됩니다. 앱에서는 **문서 추출 검증** 버튼, 각 답변 아래 **검색·답변 검증 결과 보기**, 왼쪽 **최근 전체 검증 결과**에서 확인할 수 있습니다. 자동 평가는 모델 검토이며, HTML 보고서의 실제 답변과 원문 근거를 함께 읽어 확인하세요.

```powershell
# 인용 검증과 API 키가 없는 화면 검사 (API 비용 없음)
uv run --locked --no-editable python -W error scripts/test_app.py

# 전체 PDF 임베딩, Streamlit 질문/출처 표시, 문서 밖 질문 검사 (API 비용 발생)
uv run --locked --no-editable python -W error scripts/test_app.py --live
```

구현 참고: [LangChain InMemoryVectorStore](https://reference.langchain.com/python/langchain-core/vectorstores/in_memory/InMemoryVectorStore), [OpenAI gpt-4o-mini](https://developers.openai.com/api/docs/models/gpt-4o-mini), [OpenAI text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small).

근거 인용은 모델이 문장을 다시 쓰는 대신 문서/문장 번호를 선택하면 코드가 원문을 가져오는 방식입니다. PDF 추출 과정의 공백·줄바꿈 차이는 허용하되, 지역명·금액 등 실제 글자가 달라진 인용과 존재하지 않는 문장 번호는 거부합니다. 지역이 다른 사례는 일반 규칙을 질문의 사실관계에 적용하며, 실제 운임 금액은 문서에 없으면 추정하지 않습니다.
