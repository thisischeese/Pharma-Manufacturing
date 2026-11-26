# 합성 지식 문서 (RCA Agent용)

RCA Agent(`notebooks/RCA_Agent_design.ipynb`)의 5 Why가 근거로 쓰는 **가상 문서**입니다. Lek 데이터에는 SOP·일탈 기록이 없으므로, 이 저장소의 배치별 RCA 결과(`notebooks/rca_batches/`)를 바탕으로 만들었습니다. 범위는 제품 17·23의 수율 이상·일탈 배치 23개입니다.

| 파일 | 내용 | 만든 방법 |
|---|---|---|
| `sop/*.md` | SOP 7종 — 타정 셋업(TAB-001), 정지·재시작(TAB-002), 금형·펀치(ENG-010), 교대 근무(PRD-003), API 입고 시험(QC-020), IPC 정제 무게(QC-031), 작업장 환경(FAC-005) | 배치별 RCA의 근본원인 후보가 가리키는 절차를 '현재 운영 방식'으로 서술. 근본원인이 말하는 기준(초기 설정값 표, 셋업 완료 판정 기준, 재시작 안정화 확인 등)은 **들어 있지 않음** → Why 5 '기준 부재'의 근거가 됨 |
| `deviation_records.json` | 과거 일탈 기록 23건 | `notebooks/rca_batches/build_synthetic_docs.py` 가 배치마다 RCA 결과(관찰 가설·채택 근본원인·보류)로 생성 |

## 사용 규칙

- 과거 일탈 기록은 **대상 배치보다 먼저 시작한 배치의 기록만** 검색합니다. 자기 배치와 이후 배치의 기록이 근거로 쓰이면 결론이 새어 들어갑니다.
- CAPA 상태는 모두 '계획 수립, 미이행'입니다. SOP가 아직 개정되지 않은 상태와 맞추기 위해서입니다. CAPA를 '완료'로 바꾸면 해당 SOP에도 기준을 추가해야 앞뒤가 맞습니다.
- 보고자·CAPA 조치·상태는 데이터에 없는 [합성] 항목입니다.

## 다시 만들기

```bash
cd notebooks/rca_batches
python build_synthetic_docs.py      # deviation_records.json 재생성 (SOP는 손으로 쓴 문서)
```
