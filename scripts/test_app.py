"""로컬 검증: uv run --no-editable python -W error scripts/test_app.py
실제 API/UI 검증: 위 명령 뒤에 --live 추가 (OpenAI 비용 발생)
"""

import sys
import asyncio
import gc
import logging
import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

warnings.simplefilter("error")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from openai import AsyncOpenAI, OpenAI
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest


class TestContextFilter(logging.Filter):
    def filter(self, record):
        # AppTest 생성 시 메인 스레드에는 브라우저 컨텍스트가 없습니다.
        # 이 테스트 도구 안내만 제외하며 앱/라이브러리의 다른 경고는 유지합니다.
        return not (record.threadName == "MainThread" and "missing ScriptRunContext" in record.getMessage())


logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(TestContextFilter())
REPORT_ROOT = app.ROOT
COMMUTE_QUESTION = "서울 거주·세종 근무자가 서울에서 대구로 바로 출장 가면 운임은 어떻게 지급되나요?"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def local_checks():
    # 허위 출처, 변형된 인용, 근거가 없는 답변을 차단하는지 검사합니다.
    doc = Document(page_content="국내 출장의 일비는 문서에 정해진 기준에 따라 지급합니다.", metadata={"source": "test.pdf", "page": 1})
    good = app.GroundedAnswer(supported=True, claims=[app.Claim(
        text="문서 기준에 따라 지급합니다.", evidence=[app.Evidence(document_id=1, quote=doc.page_content)]
    )])
    check(bool(app.validate_answer(good, [doc])["sources"]), "Valid evidence rejected")
    # PDF 줄바꿈/공백만 다른 인용은 허용하고, 원문의 실제 문구를 돌려줍니다.
    pdf_text = "서울(거주지)에서 대구\n(목적지)까지의 운임을 지급하여야 함."
    quote = "서울(거주지)에서 대구(목적지)까지의 운임을 지급하여야 함."
    check(app.match_original_quote(quote, pdf_text) == app.normalize(pdf_text), "PDF whitespace mismatch rejected")
    check(app.match_original_quote(quote.replace("대구", "부산"), pdf_text) is None, "Altered city accepted")
    check(app.match_original_quote("지급액은 30000원입니다.", "지급액은 20000원입니다.") is None, "Altered amount accepted")
    prepared = app.prepare_evidence({"context": json.dumps([{"document_id": 1, "text": doc.page_content}]), "question": "test"})
    prepared["generated"] = app.ReferencedAnswer(supported=True, claims=[app.ReferencedClaim(
        text="문서 기준에 따라 지급합니다.", evidence=[app.SentenceReference(document_id=1, sentence_id=1)]
    )])
    check(app.resolve_evidence(prepared).claims[0].evidence[0].quote == doc.page_content, "Original sentence not used")
    prepared["generated"].claims[0].evidence[0].sentence_id = 99
    check(not app.resolve_evidence(prepared).supported, "Invalid sentence ID accepted")
    for bad in (
        app.GroundedAnswer(supported=False, claims=[]),
        app.GroundedAnswer(supported=True, claims=[app.Claim(text="근거 없음", evidence=[])]),
        app.GroundedAnswer(supported=True, claims=[app.Claim(text="허위 출처", evidence=[app.Evidence(document_id=99, quote=doc.page_content)])]),
        app.GroundedAnswer(supported=True, claims=[app.Claim(text="변조된 인용", evidence=[app.Evidence(document_id=1, quote="문서에 존재하지 않는 가짜 근거 문장입니다.")])]),
    ):
        check(app.validate_answer(bad, [doc])["answer"] == app.NO_ANSWER, "Unsupported answer accepted")
    with patch.object(app, "ROOT", Path("missing-env-test")):
        check(app.read_api_key() == "", "Missing .env not handled")
    # 실제 Streamlit 실행기로 키가 없는 초기 화면도 검사합니다.
    with TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)), patch.object(app, "read_api_key", return_value=""):
        at = AppTest.from_string("import app\napp.main()").run(timeout=30)
        check(not at.exception, "Missing-key UI crashed")
        check(len(at.warning) == 1, "Missing-key notice missing")
    print("PASS: citation validation, fail-closed answers, missing-key Streamlit UI")


def live_checks():
    if not app.read_api_key():
        print("SKIP: .env has no OPENAI_API_KEY; live API not tested")
        return
    # Streamlit 앱을 실제로 실행하고 버튼 및 채팅 입력을 조작합니다.
    # 임시 대화 기록을 사용하므로 실제 사용자의 대화는 변경하지 않습니다.
    api_key = app.read_api_key()
    with TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)), patch.object(app, "read_api_key", return_value=api_key):
        live_ui_checks()


def live_ui_checks():
    at = AppTest.from_string("import app\napp.main()", default_timeout=180).run()
    check(not at.exception, "Initial UI crashed")
    at.button(key="read_documents").click().run(timeout=180)
    check(not at.exception and not at.error, "Live indexing failed")
    check("store" in at.session_state, "Index missing")
    print(f"PASS: all files indexed, pages={at.session_state['pages']}, chunks={at.session_state['chunks']}")
    at.chat_input[0].set_value(COMMUTE_QUESTION).run(timeout=120)
    check(not at.exception and not at.error, "Commute question UI failed")
    commute = at.session_state["messages"][-1]
    check(bool(commute["sources"]) and "세종" in commute["answer"] and "초과" in commute["answer"], "Commute rule not applied")
    check("대전" not in commute["answer"], "Document example workplace used as question workplace")
    check(any(source["file"] == "공무원여비100문100답.pdf" and source["page"] == 24 for source in commute["sources"]), "Q&A 28 citation missing")
    directory = REPORT_ROOT / ".chatbot"
    directory.mkdir(exist_ok=True)
    (directory / "commute_ui_result.json").write_text(json.dumps({"question": COMMUTE_QUESTION, "result": commute}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: exact commute question answered in Streamlit with original citations and Sejong cap")
    at.chat_input[0].set_value("국내 출장의 일비 지급 기준은 무엇인가요?").run(timeout=120)
    check(not at.exception and not at.error, "Live answer failed")
    result = at.session_state["messages"][-1]
    check(result["role"] == "assistant" and bool(result["sources"]), "Grounded answer missing")
    check(len(at.text) > len(app.list_files()), "Citation UI missing")
    print(f"PASS: gpt-4o-mini grounded response, citations={len(result['sources'])}")
    at.chat_input[0].set_value("화성 최초 식민지의 설립 연도와 초대 시장 이름은 무엇인가요?").run(timeout=120)
    check(not at.exception and not at.error, "Out-of-document question failed")
    result = at.session_state["messages"][-1]
    check(result["answer"] == app.NO_ANSWER and not result["sources"], "Unrelated question not refused")
    print("PASS: unrelated question refused without fabricated evidence")


if __name__ == "__main__":
    try:
        local_checks()
        if "--live" in sys.argv:
            live_checks()
    finally:
        # 테스터 프로세스가 종료되기 전에 OpenAI HTTP 연결도 닫습니다.
        # 실제 앱은 서버가 실행되는 동안 이 연결을 재사용합니다.
        for resource in gc.get_objects():
            if isinstance(resource, OpenAI):
                resource.close()
            elif isinstance(resource, AsyncOpenAI):
                asyncio.run(resource.close())
        # 테스트 과정에서 라이브러리가 만든 임시 폴더를 명시적으로 정리합니다.
        # 종료 시 자동 정리가 ResourceWarning을 내지 않도록 합니다.
        for resource in gc.get_objects():
            if isinstance(resource, TemporaryDirectory):
                resource.cleanup()
