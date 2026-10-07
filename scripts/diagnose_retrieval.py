"""거주지 출발 운임 질문의 검색 및 인용 검증 결과를 확인합니다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
sys.stdout.reconfigure(encoding="utf-8")

QUESTION = "서울 거주·세종 근무자가 서울에서 대구로 바로 출장 가면 운임은 어떻게 지급되나요?"

if __name__ == "__main__":
    store, _, _, _ = app.build_index(app.list_files(), app.read_api_key())
    docs = store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20}).invoke(QUESTION)
    context = json.dumps([
        {"document_id": i, "source": doc.metadata["source"], "page": doc.metadata.get("page"), "text": doc.page_content}
        for i, doc in enumerate(docs, 1)
    ], ensure_ascii=False)
    chain = app.make_chain(app.read_api_key())
    raw = chain.invoke({"context": context, "question": QUESTION})
    report = {"question": QUESTION, "retrieved": json.loads(context), "raw": raw.model_dump(), "validated": app.validate_answer(raw, docs)}
    directory = app.ROOT / ".chatbot"
    directory.mkdir(exist_ok=True)
    (directory / "retrieval_after.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 진단 스크립트에서 생성한 API 연결은 테스트 종료 시 정리합니다.
    import asyncio
    import gc
    from openai import OpenAI, AsyncOpenAI
    for resource in gc.get_objects():
        if isinstance(resource, OpenAI):
            resource.close()
        elif isinstance(resource, AsyncOpenAI):
            asyncio.run(resource.close())
