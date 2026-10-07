"""API 미사용: 구조 보존, 반대 조건 제외, 재생성/보류 제어를 검사합니다."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from rag_search import RAGIndex, split_documents
from langchain_core.documents import Document


class Stub:
    def __init__(self, replies):
        self.replies = iter(replies)

    def invoke(self, payload):
        return next(self.replies)


def main():
    pages = [Document(page_content="Q&A 1 근무지 내 차량 임차 질문?\n• 원칙적인 지급 방법입니다.", metadata={"source": "test.pdf", "page": 1}),
             Document(page_content="• 다만 예외는 별도 승인이 필요합니다.\nQ&A 2 근무지 외 차량 임차 질문?\n• 별도의 감액 방식입니다.", metadata={"source": "test.pdf", "page": 2})]
    chunks = split_documents(pages)
    assert len(chunks) == 2 and "별도 승인" in chunks[0].page_content
    assert chunks[0].metadata["page"] == 1 and chunks[0].metadata["page_end"] == 2
    vector = SimpleNamespace(similarity_search=lambda question, k: chunks)
    store = RAGIndex(vector, chunks, pages)
    assert all(doc.metadata["qa_number"] != 2 for doc in store.search("근무지 내 차량 임차"))
    assert all(doc.metadata["qa_number"] != 1 for doc in store.search("근무지 외 차량 임차"))

    doc = chunks[0]
    evidence = app.Evidence(document_id=1, quote="원칙적인 지급 방법입니다.")
    response = app.GroundedAnswer(supported=True, claims=[app.Claim(text="별도 승인이 필요한 예외를 구분합니다.", evidence=[evidence])])
    index = SimpleNamespace(search=lambda question: [doc])
    fail = app.AnswerCheck(can_answer=True, supported=False, complete=False, feedback="조건을 수정하세요.")
    passed = app.AnswerCheck(can_answer=True, supported=True, complete=True, feedback="")
    def chain(checks):
        return SimpleNamespace(ranker=Stub([app.SearchSelection(document_ids=[1], reason="관련 근거")]),
            planner=Stub([app.AnswerRequirements(required_points=["별도 승인"], forbidden_inferences=[])]),
            checker=Stub(checks), invoke=Stub([response, response]).invoke)
    retried = app.answer_question(index, chain([fail, passed]), "승인 조건?")
    assert retried["validation"]["passed"] and retried["validation"]["attempts"] == 2
    refused = app.answer_question(index, chain([fail, fail]), "승인 조건?")
    assert not refused["sources"] and not refused["validation"]["passed"]
    print("PASS: cross-page Q&A and exception preserved")
    print("PASS: opposite within/outside conditions excluded")
    print("PASS: failed verification retries once then withholds answer")


if __name__ == "__main__":
    main()
