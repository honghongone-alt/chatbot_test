"""API를 호출하지 않고 대화 저장, 복원, 초기화 화면을 검사합니다."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from streamlit.testing.v1 import AppTest
from test_app import TestContextFilter
import logging

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(TestContextFilter())


def run_checks():
    # 임시 폴더만 사용하므로 실제 사용자의 대화 기록에는 손대지 않습니다.
    with TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)), \
            patch.object(app, "read_api_key", return_value="test-key"), \
            patch.object(app, "build_index", return_value=(object(), 154, 215, [])), \
            patch.object(app, "make_chain", return_value=object()), \
            patch.object(app, "answer_question", return_value={
                "answer": "문서에 근거한 테스트 답변", "sources": [{
                    "claim": 1, "file": "test.pdf", "page": 3, "quote": "문서에서 가져온 근거 문장입니다."
                }]
            }):
        at = AppTest.from_string("import app\napp.main()", default_timeout=30).run()
        assert not at.exception
        at.button(key="read_documents").click().run()
        at.chat_input[0].set_value("저장할 질문").run()
        assert not at.exception and not at.error
        saved = app.load_history()
        assert len(saved) == 2 and saved[1]["sources"][0]["page"] == 3

        # 문서를 다시 읽거나 API 키가 바뀌어도 대화는 그대로 남습니다.
        at.button(key="read_documents").click().run()
        assert app.load_history() == saved
        with patch.object(app, "fingerprint", return_value="changed-key-or-documents"):
            at.run()
            assert at.session_state["messages"] == saved
        fresh = AppTest.from_string("import app\napp.main()", default_timeout=30).run()
        assert not fresh.exception and len(fresh.chat_message) == 2
        assert fresh.session_state["messages"] == saved
        assert any("근거 문장" in item.value for item in fresh.text)

        # 초기화는 저장 기록만 지우며, 다음 세션에서도 빈 기록을 복원합니다.
        fresh.button(key="read_documents").click().run()
        fresh.button(key="reset_history").click().run()
        assert not fresh.exception and app.load_history() == []
        assert "store" in fresh.session_state
        another = AppTest.from_string("import app\napp.main()", default_timeout=30).run()
        assert not another.exception and len(another.chat_message) == 0
        print("PASS: chat/citations saved and restored in new sessions")
        print("PASS: document reload and key changes preserve chat")
        print("PASS: reset clears disk history and keeps the vector index")


if __name__ == "__main__":
    try:
        run_checks()
    finally:
        import gc
        for resource in gc.get_objects():
            if isinstance(resource, TemporaryDirectory):
                resource.cleanup()
