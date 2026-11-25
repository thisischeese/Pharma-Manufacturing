# 배치별 수율 이상·일탈 RCA (제품 17·23)

제품 17·23에서 수율 이상(경고) 또는 수율 일탈이 판정된 **23개 배치**의 RCA 노트북입니다.
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

| 제품 | 배치 | 수율(%) | 판정 | 추적 가능 | 관찰 가설 | 채택 가설 | 채택 근본원인 | 보류 | 노트북 |
|---|---|---|---|---|---|---|---|---|---|
| 17 | 789 | 87.97 | 수율 일탈(하한 미달) | 38% | 4 | MAN-3, MAC-2 | R1 RC-MAC / R2 RC-MAN-B | — | [열기](RCA_product17_batch789.ipynb) |
| 17 | 596 | 94.99 | 수율 일탈(하한 미달) | 47% | 2 | — | 없음 | RC-MET-B | [열기](RCA_product17_batch596.ipynb) |
| 17 | 375 | 95.98 | 수율 일탈(하한 미달) | 80% | 5 | MAN-2, MAC-1 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product17_batch375.ipynb) |
| 17 | 611 | 96.01 | 수율 일탈(하한 미달) | 54% | 3 | MET-2, MEA-2 | R1 RC-MET-A / R2 RC-MEA | — | [열기](RCA_product17_batch611.ipynb) |
| 17 | 370 | 96.69 | 수율 이상(경고) | 74% | 4 | MET-2 | R1 RC-MET-A | — | [열기](RCA_product17_batch370.ipynb) |
| 17 | 717 | 96.69 | 수율 이상(경고) | 48% | 1 | — | 없음 | — | [열기](RCA_product17_batch717.ipynb) |
| 17 | 612 | 96.84 | 수율 이상(경고) | 51% | 2 | MET-2 | R1 RC-MET-A | — | [열기](RCA_product17_batch612.ipynb) |
| 17 | 618 | 96.93 | 수율 이상(경고) | 42% | 2 | — | 없음 | — | [열기](RCA_product17_batch618.ipynb) |
| 17 | 674 | 97.02 | 수율 이상(경고) | 29% | 4 | MAN-2, MAC-1, MAC-2 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product17_batch674.ipynb) |
| 17 | 617 | 97.07 | 수율 이상(경고) | 41% | 2 | — | 없음 | — | [열기](RCA_product17_batch617.ipynb) |
| 17 | 956 | 97.11 | 수율 이상(경고) | 87% | 4 | MAN-2, MAC-1, MAC-2 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product17_batch956.ipynb) |
| 17 | 591 | 97.16 | 수율 이상(경고) | 86% | 4 | MAN-2, MAC-1, MAC-2 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product17_batch591.ipynb) |
| 17 | 580 | 97.34 | 수율 이상(경고) | 89% | 4 | MAN-2, MAC-1, MAC-2 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product17_batch580.ipynb) |
| 17 | 613 | 97.34 | 수율 이상(경고) | 36% | 2 | — | 없음 | — | [열기](RCA_product17_batch613.ipynb) |
| 23 | 790 | 93.8 | 수율 일탈(하한 미달) | 66% | 9 | MAC-1, MAC-2, MET-1, MET-2 | R1 RC-MET-A / R2 RC-MAC | RC-MEA | [열기](RCA_product23_batch790.ipynb) |
| 23 | 729 | 95.22 | 수율 일탈(하한 미달) | 58% | 4 | MAN-2, MAC-1 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product23_batch729.ipynb) |
| 23 | 393 | 95.5 | 수율 일탈(하한 미달) | 85% | 2 | — | 없음 | — | [열기](RCA_product23_batch393.ipynb) |
| 23 | 564 | 96.28 | 수율 이상(경고) | 68% | 5 | MAN-2, MAC-1, MAC-2 | R1 RC-MAC / R2 RC-MAN-A | RC-MET-B, RC-ENV | [열기](RCA_product23_batch564.ipynb) |
| 23 | 903 | 96.31 | 수율 이상(경고) | 66% | 1 | — | 없음 | RC-ENV | [열기](RCA_product23_batch903.ipynb) |
| 23 | 889 | 96.5 | 수율 이상(경고) | 40% | 6 | MAN-2, MAC-1, MAC-2, MAC-3, MET-1 | R1 RC-MET-A / R2 RC-MAC / R3 RC-MAN-A | RC-ENV | [열기](RCA_product23_batch889.ipynb) |
| 23 | 743 | 96.69 | 수율 이상(경고) | 36% | 1 | — | 없음 | — | [열기](RCA_product23_batch743.ipynb) |
| 23 | 740 | 96.73 | 수율 이상(경고) | 91% | 5 | MAN-2, MAC-1 | R1 RC-MAC / R2 RC-MAN-A | — | [열기](RCA_product23_batch740.ipynb) |
| 23 | 394 | 96.88 | 수율 이상(경고) | 85% | 2 | MET-2 | R1 RC-MET-A | — | [열기](RCA_product23_batch394.ipynb) |

### 채택된 근본원인 빈도

| RC | 관점 | 근본원인 | 채택 배치 수 |
|---|---|---|---|
| RC-MAC | 설비 | 재시작 후 공정 조건 안정화 확인 절차와 압축부 상태 점검·경보 기준이 없다. | 11 |
| RC-MAN-A | 인력 | 야간·주말 셋업과 이상 대응에 숙련자 배치·지원 요청(에스컬레이션) 기준이 없다. | 9 |
| RC-MET-A | 방법 | 제품별로 검증된 초기 셋업 설정값(충전 깊이·압축력 레시피)이 정립되어 있지 않다. | 6 |
| RC-MAN-B | 인력 | 운전 설정값 이중 확인과 레시피 선택 검증 절차가 없다. | 1 |
| RC-MEA | 측정 | 셋업 단계의 무게 측정 체계(SREL 판정 한계·센서 확인·IPC 주기) 확인 기준이 없다. | 1 |

> 결과는 관찰 데이터의 연관성 검정입니다. 채택 근본원인의 Why 2~5 중 [추정] 단계는 SOP·셋업 기록·설정 변경 이력·정비 이력으로 확인해야 하며,
> 손실 추적 가능 비율이 50% 미만인 배치는 결론이 기록된 손실에 한정됩니다.
