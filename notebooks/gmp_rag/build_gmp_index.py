"""KGMP 고시(의약품 제조 및 품질관리에 관한 규정) → 청크 → KoBERT 임베딩 캐시.

순서 (저장소 루트 기준)
    cd notebooks/gmp_rag
    npm install            # kordoc
    npm run parse          # PDF → data/knowledge/gmp_kordoc.md
    uv run python build_gmp_index.py          # 고형제 적용 범위만 임베딩 (--all : 전체)

출력
    data/knowledge/gmp_chunks.json   청크 목록 (별표·항목·본문)
    data/knowledge/gmp_kobert.npy    embedded_ids 순서의 KoBERT 벡터 (L2 정규화)

gmp_kordoc.md 가 없으면 pdftotext(poppler)로 대신 읽고 parser 필드에 'pdftotext (대체)' 를 남김.
kordoc 결과가 생기면 이 스크립트를 다시 실행해 덮어쓰면 됨.

고형제(정제) 수율 RCA에 쓸 범위 : 본문, 별표 9(컴퓨터화 시스템), 13(적격성평가와 밸리데이션),
14(실시간 출하 시험), 16(참조·보관검체), 17(완제의약품 제조). 나머지 별표(무균·생물·방사성 등)는
청크에는 남기되 applicable=false 로 표시해 검색에서 뺌.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "knowledge"
KORDOC_MD = OUT / "gmp_kordoc.md"

ANNEX_TITLES = {
    "1": "무균의약품 제조", "2": "생물유래의약품의 원료 및 완제의약품 제조", "2의2": "첨단바이오의약품 제조",
    "3": "방사성의약품 제조", "4": "의료용 고압가스 제조", "5": "한약(생약)제제 제조", "6": "출발물질 및 포장자재 검체 채취",
    "7": "액제, 크림제 및 연고제 제조", "8": "흡입용 압축 정량 에어로솔 제제 제조", "9": "컴퓨터화 시스템",
    "10": "전리방사선의 의약품 제조 사용", "11": "임상시험용의약품 제조", "12": "사람의 혈액 또는 혈장에서 유래한 의약품 제조",
    "13": "적격성평가와 밸리데이션", "14": "실시간 출하 시험 및 매개변수기반 출하", "15": "원료의약품 제조",
    "16": "참조검체 및 보관검체", "17": "완제의약품 제조", "18": "제조소 총람 구성",
}
APPLICABLE = {"본문", "9", "13", "14", "16", "17"}
PAGE_HEADER = "의약품 제조 및 품질관리에 관한 규정"
MAX_CHARS = 900

ANNEX_RE = re.compile(r"^\s*#*\s*\[별표\s*(\d+(?:의\d+)?)\]\s*(.*)$")
ARTICLE_RE = re.compile(r"^\s*#*\s*제\d+조(?:의\d+)?\s*\(.+?\)")
ITEM_RE = re.compile(r"^\s*#*\s*\d+(?:\.\d+)*\.\s+\S")
SUB_RE = re.compile(r"^\s*([가-하]\.|\d+\)|[가-하]\))\s")


def source_pages() -> tuple[str, list[tuple[int | None, str]]]:
    """(parser 이름, [(쪽 번호 또는 None, 텍스트)])"""
    if KORDOC_MD.exists():
        return "kordoc", [(None, KORDOC_MD.read_text(encoding="utf-8"))]
    pdf = next(DATA.glob("*.pdf"))
    if not shutil.which("pdftotext"):
        sys.exit("gmp_kordoc.md 가 없고 pdftotext도 없습니다. 먼저 `npm install && npm run parse` 를 실행하세요.")
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return "pdftotext (대체)", [(i, p) for i, p in enumerate(txt.split("\f"), 1)]


def sections(pages):
    """별표 경계로 나눈 (annex, [(page, line)])"""
    cur, buf, out = "본문", [], []
    for page, text in pages:
        for line in text.splitlines():
            s = line.strip()
            if (not s or s == PAGE_HEADER or re.fullmatch(r"-?\s*\d+\s*-?", s)
                    or ("법제처" in s and "국가법령정보센터" in s) or s.startswith("[시행 20")):
                continue
            m = ANNEX_RE.match(s)
            if m and (not m.group(2) or m.group(2).strip().startswith(ANNEX_TITLES.get(m.group(1), "\0")[:4])):
                out.append((cur, buf))
                cur, buf = m.group(1), []
                continue
            buf.append((page, s))
    out.append((cur, buf))
    # 목차처럼 내용이 거의 없는 구간 제거, 같은 별표가 이어지면 합침
    merged = []
    for annex, lines in out:
        if sum(len(l) for _, l in lines) < 300:
            continue
        if merged and merged[-1][0] == annex:
            merged[-1][1].extend(lines)
        else:
            merged.append((annex, lines))
    return merged


def items(lines, article: bool = False):
    """본문은 제N조, 별표는 최상위 번호 항목(1. / 6.3.)으로 나눔"""
    head, buf, out = "(머리말)", [], []
    for page, s in lines:
        is_head = ARTICLE_RE.match(s) if article else (ITEM_RE.match(s) and len(s) < 120)
        if is_head:
            if buf:
                out.append((head, buf))
            head, buf = s.lstrip("# ").strip(), []
            continue
        buf.append((page, s))
    if buf:
        out.append((head, buf))
    return out


def pack(head, lines):
    """MAX_CHARS 이하로 하위 항목(가. 1) …) 경계에서 자름"""
    pieces, cur = [], []
    for page, s in lines:
        if cur and SUB_RE.match(s) and sum(len(x) for _, x in cur) > MAX_CHARS * 0.6:
            pieces.append(cur)
            cur = []
        cur.append((page, s))
        if sum(len(x) for _, x in cur) > MAX_CHARS * 1.4:
            pieces.append(cur)
            cur = []
    if cur:
        pieces.append(cur)
    for p in pieces:
        pg = [x for x, _ in p if x is not None]
        yield {"heading": head, "text": " ".join(s for _, s in p), "pages": [min(pg), max(pg)] if pg else None}


def build_chunks():
    parser, pages = source_pages()
    chunks = []
    for annex, lines in sections(pages):
        title = "의약품 제조 및 품질관리에 관한 규정 (본문)" if annex == "본문" else f"[별표 {annex}] {ANNEX_TITLES.get(annex, '')}"
        for n, (head, body) in enumerate(items(lines, article=annex == "본문")):
            for k, c in enumerate(pack(head, body)):
                if len(c["text"]) < 40:
                    continue
                chunks.append({"chunk_id": f"GMP-{annex}-{n:03d}-{k}", "annex": annex, "annex_title": title,
                               "applicable": annex in APPLICABLE, **c})
    return parser, chunks


def main():
    parser, chunks = build_chunks()
    OUT.mkdir(parents=True, exist_ok=True)
    # 기본은 고형제 적용 범위만 임베딩 (전체 674개는 CPU에서 수 분) — 전체가 필요하면 --all
    targets = chunks if "--all" in sys.argv else [c for c in chunks if c["applicable"]]
    meta = {"source": next(DATA.glob("*.pdf")).name, "parser": parser, "embedding": "skt/kobert-base-v1 mean pooling",
            "applicable_annexes": sorted(APPLICABLE), "n_chunks": len(chunks),
            "embedded_ids": [c["chunk_id"] for c in targets], "chunks": chunks}
    (OUT / "gmp_chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{parser}] 청크 {len(chunks)}개 (고형제 적용 {sum(c['applicable'] for c in chunks)}개)")

    sys.path.insert(0, str(Path(__file__).parent))
    from kobert_embed import KoBERTEmbedder
    emb = KoBERTEmbedder()
    import numpy as np
    X = emb.encode([f"{c['annex_title']} {c['heading']} {c['text']}" for c in targets])
    np.save(OUT / "gmp_kobert.npy", X)
    print("임베딩", X.shape, "→", OUT / "gmp_kobert.npy")


if __name__ == "__main__":
    if "--chunks-only" in sys.argv:
        p, c = build_chunks()
        print(p, len(c))
        for x in c[:3] + [y for y in c if y["annex"] == "17"][:4]:
            print(x["chunk_id"], x["heading"][:60], "|", x["text"][:160])
    else:
        main()
