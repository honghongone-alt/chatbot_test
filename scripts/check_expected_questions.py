"""문서 기반 예상 질문 10개를 실제 RAG 함수로 검사합니다 (API 비용 발생)."""

import asyncio
import gc
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("error")
import app
from openai import AsyncOpenAI, OpenAI

CASES = [
    {"question": "서울 거주·세종 근무자가 서울에서 대구로 바로 출장 가면 운임은 어떻게 지급되나요?",
     "expected": "서울→대구 운임을 지급하되 세종→대구 운임이 상한. 대전을 질문의 근무지로 바꾸면 안 됨.", "reference": "100문100답 Q&A 28, PDF 24쪽"},
    {"question": "오전 9~11시, 오후 13~17시에 근무지 내 출장을 각각 다녀왔습니다. 하루 출장여비는 3만원인가요?",
     "expected": "하루 최대 2만원이므로 3만원이 아닌 2만원 지급.", "reference": "100문100답 Q&A 10, PDF 16쪽"},
    {"question": "왕복 1.5km인 근거리 출장을 3시간 다녀왔고 부득이하게 교통비 7천원이 들었습니다. 영수증이 있으면 얼마까지 지급되나요?",
     "expected": "4시간 미만 근거리 출장은 운임 1만원 한도 내 실비이므로 증빙이 있는 7천원 지급. 정액 1만원 지급 아님.", "reference": "100문100답 Q&A 12, PDF 16쪽"},
    {"question": "왕복 2km 이내 근거리 출장인지는 지도상의 직선거리로 판단하나요?",
     "expected": "직선거리가 아닌 최단 이동거리로 판단.", "reference": "100문100답 Q&A 13, PDF 17쪽"},
    {"question": "공무상 부득이한 사유 없이 개인 자동차로 근무지 외 출장을 갔습니다. 연료비와 통행료를 받을 수 있나요?",
     "expected": "원칙은 근무지~목적지 대중교통 요금. 연료비·통행료·주차료는 공무상 부득이한 사유로 인정된 경우에 가능.", "reference": "100문100답 Q&A 21, PDF 22쪽"},
    {"question": "근무지 내 출장인데 불가피하게 숙박했습니다. 숙박비를 자동으로 받을 수 있나요?",
     "expected": "원칙 지급 불가. 특별한 사유가 인정되고 인사혁신처장·기획재정부장관과 협의한 경우 예외 가능.", "reference": "100문100답 Q&A 9, PDF 15쪽"},
    {"question": "근무지 내 출장에 렌터카를 이용했습니다. 차량 임차료를 여비 항목으로 지급할 수 있나요?",
     "expected": "여비가 아닌 다른 예산 항목(임차료 등)으로 지급. 근무지 내 출장은 별도 운임 없음; 임차차량 이용 시 1만원 감액.", "reference": "100문100답 Q&A 8, PDF 15쪽"},
    {"question": "서울에서 제주도로 출장 가는 첫날 근무지에서 김포공항까지 공용차량을 탔습니다. 첫날 일비는 어떻게 되나요?",
     "expected": "공용차량 이용일의 일비는 2분의 1 감액. 총 출장기간 전부를 감액하는 것이 아님.", "reference": "100문100답 Q&A 30, PDF 24쪽"},
    {"question": "세종 근무자가 부산으로 월~화 출장인데 금요일에 부산 자택으로 미리 가서 숙박하고 화요일에 복귀했습니다. 운임과 숙박비는 어떻게 지급하나요?",
     "expected": "출장 수행을 위한 운임 가능. 당초 출발일보다 할증된 요금 차액은 본인 부담. 자가 숙박비 지급 안 함.", "reference": "100문100답 Q&A 19, PDF 21쪽"},
    {"question": "2026년 10월 7일 서울에서 대구까지 KTX 일반실 실제 승차권 가격은 정확히 얼마인가요?",
     "expected": "문서에 특정 날짜 실제 운임이 없으므로 금액을 추측하지 않고 답변 보류.", "reference": "문서 밖 정보/추측 방지 검사", "abstain": True},
]


def main():
    if not app.read_api_key():
        raise RuntimeError(".env의 OPENAI_API_KEY를 설정하세요.")
    store, pages, chunks, notices = app.build_index(app.list_files(), app.read_api_key())
    chain = app.make_chain(app.read_api_key())
    report = {"model": "gpt-4o-mini", "embedding": "text-embedding-3-small", "pages": pages, "chunks": chunks, "notices": notices, "cases": []}
    directory = app.ROOT / ".chatbot"
    directory.mkdir(exist_ok=True)
    output = directory / "expected_questions_results.json"
    for number, case in enumerate(CASES, 1):
        try:
            result = app.answer_question(store, chain, case["question"])
            basic_pass = (result["answer"] == app.NO_ANSWER and not result["sources"]) if case.get("abstain") else (result["answer"] != app.NO_ANSWER and bool(result["sources"]))
            row = {"number": number, **case, "result": result, "basic_pass": basic_pass}
        except Exception as exc:
            row = {"number": number, **case, "error_type": type(exc).__name__, "basic_pass": False}
        report["cases"].append(row)
        # 중간 결과도 저장해 API 오류가 나더라도 완료된 검사를 확인할 수 있게 합니다.
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"REPORT: {output}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    finally:
        # 검사 후 HTTP 연결을 닫습니다. 사용자 대화 기록은 변경하지 않습니다.
        for resource in gc.get_objects():
            if isinstance(resource, OpenAI):
                resource.close()
            elif isinstance(resource, AsyncOpenAI):
                asyncio.run(resource.close())
