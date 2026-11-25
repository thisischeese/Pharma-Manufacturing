"""제품 17·23의 수율 이상·일탈 배치마다 RCA 노트북을 생성한다.

사용: uv run python notebooks/rca_batches/build_notebooks.py  (저장소 루트에서)
생성 후 실행: 각 노트북을 jupyter nbconvert --execute 로 실행 (README 참고)
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rca_core import find_data_dir, judged  # noqa: E402

PRODUCTS = [17, 23]


def targets():
    j = judged(str(find_data_dir()))
    t = j[j.code.isin(PRODUCTS) & j.판정.ne("정상")].copy()
    return t.sort_values(["code", "robust_z"])


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def notebook(batch: int, prod: int, verdict: str):
    kind = "수율 일탈" if "일탈" in verdict else "수율 이상"
    cells = [
        md(f"""
# {kind} RCA — 제품 {prod} · 배치 {batch}

| 항목 | 내용 |
|---|---|
| 판정 | {verdict} |
| 절차 | 상세 노트북 `notebooks/RCA_Lek_yield_deviation.ipynb`(배치 790)와 같은 7단계: 규격 → 5W1H → 발생 보고서 → 6M → 가설별 5 Why → 통계 검정 → RCA 보고서 |
| 코드 | 모든 계산은 같은 폴더의 `rca_core.py` 에 있고, 이 노트북은 결과와 해석을 보여줍니다 |
| 실행 | 저장소 루트에서 `uv sync` → 이 폴더에서 노트북 실행 |
| 표시 규칙 | 5 Why 근거 [데이터] / [데이터+추정] / [추정] · 데이터에 없는 보고서 항목은 [가상] |
"""),
        code(f"""
from rca_core import RCA
ctx = RCA({batch})
ctx.show_overview()
"""),
        md("""
---
## 1. STEP 1 — 수율 규격과 판정

제품별 과거 수율로 세운 통계적 관리 기준입니다. 기준선 = 중앙값, σ̂ = 1.4826 × MAD.
**경고 하한(중앙값 − 2σ̂) 미만 = 수율 이상**, **일탈 하한(중앙값 − 3σ̂) 미만 또는 100% 초과 = 수율 일탈**.
"""),
        code("ctx.show_spec()"),
        md("""
---
## 2. STEP 2 — 5W1H 문제 정의

### 2.1 생산 과정 — 대상 배치와 대표 정상 배치의 10초 시계열
"""),
        code("ctx.show_trace()"),
        md("""
### 2.2 손실 분해

손실 = 셋업 폐기 + 본생산 폐기 + 타정 이후 공정 손실 + 미기록 손실(타정기 카운터 밖 손실). '중앙값 대비'는 같은 제품 다른 배치의 중앙값과의 차이입니다.
"""),
        code("ctx.show_loss()"),
        md("### 2.3 5W1H"),
        code("ctx.show_5w1h()"),
        md("""
---
## 3. STEP 3 — 수율 이상 발생 보고서

팀 양식의 발생보고서 항목을 따릅니다. 데이터에 없는 항목(보고자, 발견 시각, 초기 조치)은 **[가상]** 예시입니다.
"""),
        code("ctx.show_incident_report()"),
        md("""
---
## 4. STEP 4 — 6M 원인 후보

6M 관점별 1~3개, 총 14개 가설을 같은 기준으로 봅니다. 대상 배치의 지표가 같은 제품 참조 배치와 비교해
가설 방향으로 **상위·하위 5% 안**이면(예/아니오 지표는 '해당'이면) 관찰된 것으로 보고 5 Why로 넘깁니다.
관찰되지 않은 가설은 이 배치를 설명할 수 없으므로 Why 1에서 종결합니다.
"""),
        code("ctx.show_hypotheses()\nctx.draw_fishbone(after=False)"),
        md("""
---
## 5. STEP 5 — 가설별 5 Why

- Why 1 질문은 모든 경로에 공통("왜 수율이 떨어졌나?"), **Why 1의 답 = 가설**입니다. 가설이 작용한 손실 구간은 '작용 위치'로 따로 적습니다.
- 각 답의 근거: **[데이터]** 데이터로 관찰한 사실 · **[데이터+추정]** 사실은 관찰되지만 이유·방향은 해석 · **[추정]** 데이터에 근거 없음(현장 확인 필요).
- 같은 Why 5(근본원인)에 도달한 경로는 하나의 근본원인 후보(RC)로 묶습니다.
"""),
        code("ctx.show_five_why()"),
        md("### 5.1 근본원인 후보 통합"),
        code("ctx.show_rc_table()"),
        md("""
---
## 6. STEP 6 — 통계 검정

| 단계 | 질문 | 방법 | 판정 |
|---|---|---|---|
| ① 대상 배치 해당성 | 이 조건이 대상 배치에 실제로 있었나? | 경험적 p (참조 중 대상만큼 극단인 비율) | p < 0.05 또는 '해당' |
| ② 수율 연관성 | 평소에도 이 조건이 수율 저하와 함께 나타나나? | 스피어만 ρ / 맨-휘트니 U (**양측**), Holm 보정 | Holm p < 0.05 **이고 방향이 가설과 일치** → 채택 |
| ② 보류 규칙 | 비교할 비슷한 사례가 있나? 지표가 조건 자체를 재나? | 참조 중 대상 수준의 절반 이상 극단인 배치 수 · 대리 지표 여부 | 비슷한 사례 3개 미만이고 유의하지 않으면 「보류(참조 부족)」, 대리 지표(계절)는 유의해도 「보류(대리 지표)」 |
| ③ 근본원인 확인 | 같은 현상을 두 번 센 것은 아닌가? | 합류 고리 상관, RC 간 상관, 편 스피어만(다른 RC 통제) | 독립 효과가 유의하고 방향 일치 → RC 채택 |

검정 방법별 목적과 필요성은 팀 문서 「RCA 가설 통계 검정 방법 정리」를 참고하세요.
"""),
        code("ctx.show_tests()"),
        md("### 6.1 근본원인 확인과 최종 판정"),
        code("ctx.show_stage3()"),
        md("""
---
## 7. STEP 7 — RCA 결과 정리

### 7.1 근본원인 해소 시 추정 수율 (단순 산술)
"""),
        code("ctx.show_whatif()"),
        md("### 7.2 특성요인도 (검정 후)"),
        code("ctx.draw_fishbone(after=True)"),
        md("### 7.3 근본원인 조사 보고서 (RCA)"),
        code("ctx.show_rca_report()"),
        md("### 7.4 결론"),
        code("ctx.show_conclusion()"),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.12"}}
    return nb


def write_notebooks():
    t = targets()
    for b, r in t.iterrows():
        path = HERE / f"RCA_product{int(r.code)}_batch{b}.ipynb"
        nbf.write(notebook(int(b), int(r.code), r.판정), path)
        print("written", path.name, r.판정)


def write_readme():
    """배치별 결과 요약표를 README.md 로 쓴다 (노트북과 같은 계산)."""
    import pandas as pd
    from rca_core import RCA, RC
    rows = []
    for b, r in targets().iterrows():
        x = RCA(int(b))
        s = x.summary()
        s["노트북"] = f"[열기](RCA_product{int(r.code)}_batch{b}.ipynb)"
        rows.append(s)
    S = pd.DataFrame(rows)
    S = S[["제품", "배치", "수율(%)", "판정", "추적 가능", "관찰 가설", "채택 가설", "채택 근본원인", "보류", "노트북"]]
    S["채택 가설"] = S["채택 가설"].replace("", "—")
    cnt = {}
    for v in S["채택 근본원인"]:
        for part in v.split(" / "):
            rc = part.split(" ")[-1]
            if rc in RC:
                cnt[rc] = cnt.get(rc, 0) + 1
    table = "\n".join(["| " + " | ".join(S.columns) + " |", "|" + "---|" * len(S.columns)]
                      + ["| " + " | ".join(str(v) for v in row) + " |" for row in S.values])
    freq = "\n".join(f"| {rc} | {RC[rc][0]} | {RC[rc][1]} | {n} |" for rc, n in sorted(cnt.items(), key=lambda kv: -kv[1]))
    text = f"""# 배치별 수율 이상·일탈 RCA (제품 17·23)

제품 17·23에서 수율 이상(경고) 또는 수율 일탈이 판정된 **{len(S)}개 배치**의 RCA 노트북입니다.
절차는 상세 노트북 `../RCA_Lek_yield_deviation.ipynb`(배치 790)와 같은 7단계이며, 계산은 모두 `rca_core.py` 에 있습니다.

## 실행

```bash
uv sync                                                       # 저장소 루트에서
uv run python notebooks/rca_batches/build_notebooks.py           # 노트북 생성 (실행 결과 없음)
cd notebooks/rca_batches && uv run jupyter nbconvert --to notebook --execute --inplace RCA_product*.ipynb
uv run python build_notebooks.py --readme                     # 요약표 갱신
```

## 상세 노트북(배치 790) 대비 바뀐 규칙

| 항목 | 상세 노트북 | 배치별 노트북 |
|---|---|---|
| 5 Why 근거 표시 | [데이터] / [추정] | **[데이터] / [데이터+추정] / [추정]** — 사실은 관찰되지만 이유·방향이 해석인 답을 구분 |
| 단계 ② 검정 | 단측검정 | **양측검정 + 방향 일치** 요구 (반대 방향 유의는 '기각(반대 방향)'으로 따로 표시) |
| 단계 ② 기각 조건 | 유의하지 않으면 기각 | 대상 수준과 비슷한 참조 배치가 3개 미만이면 **'보류(참조 부족)'** — 비교할 사례가 없을 때는 기각하지 않음 |
| 대리 지표 | 계절 가설도 채택 가능 | 계절(ENV-1)은 유의해도 **'보류(대리 지표)'** — 온·습도 자체를 재지 않으므로 |
| 가설 | 12개 | 14개: **MAN-3 운전 설정값 이탈**(배치 789 같은 설정 이상), **MET-3 셋업 직후 조건 재조정**(셋업 완료 판정 기준 부재) 추가 |
| 근본원인 | RC-MET 하나 | **RC-MET-A** 검증된 초기 설정값 부재 / **RC-MET-B** 셋업 완료 판정 기준 부재로 분리 |
| 날짜 보정 | 다음 배치와의 연속성으로 수동 확인 | 제품 파일 전체에서 '배치 번호 순서 = 시간 순서'를 지키는 해석을 동적계획법으로 선택 |

이 규칙 때문에 배치 790의 결과가 상세 노트북과 한 곳 다릅니다: MEA-1(셋업 중 SREL)이 '기각' → **'보류(참조 부족)'** (비슷한 수준의 참조 배치가 2개뿐).

## 결과 요약

{table}

### 채택된 근본원인 빈도

| RC | 관점 | 근본원인 | 채택 배치 수 |
|---|---|---|---|
{freq}

> 결과는 관찰 데이터의 연관성 검정입니다. 채택 근본원인의 Why 2~5 중 [추정] 단계는 SOP·셋업 기록·설정 변경 이력·정비 이력으로 확인해야 하며,
> 손실 추적 가능 비율이 50% 미만인 배치는 결론이 기록된 손실에 한정됩니다.
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")
    print("README.md written")
    return S


if __name__ == "__main__":
    if "--readme" in sys.argv:
        write_readme()          # 실행된 노트북은 건드리지 않음
    else:
        write_notebooks()
