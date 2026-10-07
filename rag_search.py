"""문서 구조를 보존하는 분할과 키워드/벡터 결합 검색."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_VERSION = "structured-hybrid-3"


def split_documents(pages: list[Document]) -> list[Document]:
    # Q&A는 질문부터 다음 질문 직전까지 함께 보존합니다. 페이지를 넘어도 연결합니다.
    units = []
    grouped = {}
    for page in pages:
        grouped.setdefault(page.metadata["source"], []).append(page)
    qa_pattern = re.compile(r"^\s*Q&A\s+(\d+)\s+([^\n]+)", re.MULTILINE)
    for source, source_pages in grouped.items():
        current = None
        scope = None
        for page in source_pages:
            # 규정의 장 제목을 이후 페이지에도 이어 붙여 내/외 출장 혼동을 막습니다.
            heading_matches = re.findall(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧ]+\.\s*([^\n]+)", page.page_content, re.MULTILINE)
            for heading in heading_matches:
                compact_heading = re.sub(r"\s+", "", heading)
                scope = "within" if "근무지내" in compact_heading else "outside" if "근무지외" in compact_heading else None
            page = page.model_copy(deep=True)
            page.metadata["scope"] = scope
            matches = list(qa_pattern.finditer(page.page_content))
            if not matches:
                if current is not None:
                    current.page_content += "\n" + page.page_content
                    current.metadata["page_end"] = page.metadata["page"]
                else:
                    units.append(page.model_copy(deep=True))
                continue
            prefix = page.page_content[:matches[0].start()]
            if current is not None and prefix.strip():
                current.page_content += "\n" + prefix
                current.metadata["page_end"] = page.metadata["page"]
            elif prefix.strip():
                units.append(Document(page_content=prefix, metadata=dict(page.metadata)))
            for i, match in enumerate(matches):
                if current is not None:
                    units.append(current)
                end = matches[i + 1].start() if i + 1 < len(matches) else len(page.page_content)
                current = Document(page_content=page.page_content[match.start():end], metadata={
                    **page.metadata, "qa_number": int(match.group(1)), "title": match.group(2).strip(),
                    "page_end": page.metadata["page"], "unit_type": "qa",
                })
        if current is not None:
            units.append(current)
    # 규정은 문단 경계를 우선합니다. 긴 Q&A만 보조적으로 분할합니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for unit in units:
        for part in splitter.split_documents([unit]):
            part.metadata.setdefault("unit_type", "regulation")
            headings = re.findall(r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧ]+[.\s]|\d+[.]\s|[가-하][.]\s)[^\n]+", unit.page_content, re.MULTILINE)
            part.metadata.setdefault("title", " / ".join(h.strip() for h in headings[:3]))
            chunks.append(part)
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = number
    return chunks


def terms(text: str) -> list[str]:
    # 한국어 조사를 포함한 표현 차이를 보완하도록 단어와 두 글자 묶음을 함께 사용합니다.
    aliases = {"렌터카": " 렌트차량 차량 임차 ", "자택": " 자가 숙박 ", "자가용": " 개인 자동차 ", "교통비": " 운임 "}
    for key, value in aliases.items():
        text = text.replace(key, key + value)
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text.lower())
    tokens = list(words)
    for word in words:
        if len(word) >= 3 and re.fullmatch(r"[가-힣]+", word):
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


@dataclass
class RAGIndex:
    vector_store: InMemoryVectorStore
    chunks: list[Document]
    pages: list[Document]

    def __post_init__(self):
        self.counts = [Counter(terms(chunk.page_content)) for chunk in self.chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.frequency = Counter(term for count in self.counts for term in count)

    @property
    def embedding(self):
        return self.vector_store.embedding

    def as_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)

    def search(self, question: str, limit: int = 16) -> list[Document]:
        query_terms = set(terms(question))
        scores = []
        for i, count in enumerate(self.counts):
            score = 0.0
            for term in query_terms:
                frequency = count.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.chunks) - self.frequency[term] + .5) / (self.frequency[term] + .5))
                denominator = frequency + 1.5 * (.25 + .75 * self.lengths[i] / max(self.average_length, 1))
                score += idf * frequency * 2.5 / denominator
            scores.append(score)
        lexical = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:30]
        semantic = self.vector_store.similarity_search(question, k=min(30, len(self.chunks)))
        # 순위를 통합해 키워드 일치와 의미의 가까움을 함께 이용합니다.
        fused = Counter()
        for rank, i in enumerate(lexical, 1):
            if scores[i] > 0:
                fused[i] += .6 / (40 + rank)
        for rank, doc in enumerate(semantic, 1):
            fused[doc.metadata["chunk_id"]] += .4 / (40 + rank)
        ranked = [self.chunks[i] for i, _ in fused.most_common()]
        query_compact = re.sub(r"\s+", "", question)
        within = "근무지내" in query_compact
        outside = "근무지외" in query_compact
        filtered = []
        for doc in ranked:
            # 내/외 구분이 명시된 Q&A를 반대 조건의 질문에 적용하지 않습니다.
            heading = re.sub(r"\s+", "", doc.page_content.split("•", 1)[0]) if doc.metadata.get("unit_type") == "qa" else ""
            if within != outside:
                if within and doc.metadata.get("scope") == "outside":
                    continue
                if outside and doc.metadata.get("scope") == "within":
                    continue
                if within and "근무지외" in heading and "근무지내" not in heading:
                    continue
                if outside and "근무지내" in heading and "근무지외" not in heading:
                    continue
            filtered.append(doc)
        return filtered[:limit]


def extraction_report(pages: list[Document], chunks: list[Document], notices: list[str]) -> dict:
    files = {}
    for page in pages:
        entry = files.setdefault(page.metadata["source"], {"pages": 0, "characters": 0, "qa_numbers": []})
        entry["pages"] += 1
        entry["characters"] += len(page.page_content)
    for chunk in chunks:
        number = chunk.metadata.get("qa_number")
        if number is not None:
            files[chunk.metadata["source"]]["qa_numbers"].append(number)
    for entry in files.values():
        entry["qa_numbers"] = sorted(set(entry["qa_numbers"]))
    return {"files": files, "pages": len(pages), "chunks": len(chunks), "notices": notices}
