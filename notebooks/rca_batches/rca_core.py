"""배치별 수율 이상·일탈 RCA 공통 모듈.

`notebooks/RCA_Lek_yield_deviation.ipynb`(배치 790 상세 노트북)의 절차를 함수로 묶은 것입니다.
배치 노트북은 `ctx = RCA(배치번호)` 한 줄로 모든 계산을 끝내고, 각 단계에서 `ctx.show_*()` 로 결과를 봅니다.

상세 노트북 대비 달라진 점 (팀 리뷰 반영)
- 5 Why 근거 표시 3단계: [데이터] / [데이터+추정] / [추정]
- 단계 ② 수율 연관성: 양측검정 + "효과 방향이 가설과 일치"를 함께 요구
- 단계 ② 기각 조건 강화: 참조 배치 중 대상 배치와 비슷한 수준의 조건이 3개 미만이면 '기각'이 아니라 '보류(현장 확인)'
- 가설 추가: MAN-3 운전 설정값 이탈, MET-3 셋업 직후 조건 재조정(셋업 완료 판정 기준 부재)
- 근본원인(RC) 분리: '검증된 초기 설정값 부재'(RC-MET-A)와 '셋업 완료 판정 기준 부재'(RC-MET-B)
"""
from __future__ import annotations

import re
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import koreanize_matplotlib  # noqa: F401
from scipy import stats
from IPython.display import HTML, Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
import logging  # noqa: E402
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

# ════════════════════════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════════════════════════
SIGMA_ALERT, SIGMA_ACTION, UPPER_LIMIT, MIN_BATCHES = 2, 3, 100.0, 10
ALPHA = 0.05
MIN_NEAR = 3            # 단계 ② '기각'에 필요한, 대상 배치와 비슷한 수준의 참조 배치 수
SAMPLE_SEC = 10         # 시계열 샘플 간격(초)
LONG_STOP = 6           # 1분 = 6샘플
EARLY_WIN = 360         # 셋업 직후 60분 = 360샘플

C_NORMAL, C_LOW, C_GRAY, C_TEXT2 = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C_TEXT2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "xtick.color": C_TEXT2, "ytick.color": C_TEXT2, "legend.frameon": False,
    "axes.unicode_minus": False, "axes.axisbelow": True,
})

SLO_MONTH = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6,
             "jul": 7, "avg": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12}
SEASON = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
          6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}


# ════════════════════════════════════════════════════════════════════
# 데이터 로드 (EDA 3장과 같은 정리)
# ════════════════════════════════════════════════════════════════════
def find_data_dir() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "Laboratory.csv").exists():
            return p / "data"
    raise FileNotFoundError("data/Laboratory.csv 를 찾을 수 없습니다.")


@lru_cache(maxsize=None)
def load_base(data_dir: str):
    d = Path(data_dir)
    lab = pd.read_csv(d / "Laboratory.csv", sep=";")
    proc = pd.read_csv(d / "Process.csv", sep=";")
    norm = pd.read_csv(d / "Normalization.csv", sep=";").set_index("Product code")["Normalisation factor"]
    for c in ["api_water", "api_total_impurities", "api_l_impurity", "api_ps01", "api_ps05", "api_ps09"]:
        lab[c] = pd.to_numeric(lab[c].astype(str).str.strip().replace("", np.nan), errors="coerce")
    dup = ["Drug release average (%)", "Drug release min (%)", "Residual solvent",
           "Total impurities", "Impurity O", "Impurity L"]
    df = lab.set_index("batch").join(proc.set_index("batch").drop(columns=["code"] + dup))
    mon, yr = df["start"].str.split(".", expand=True).T.values
    df["start_month"] = pd.to_datetime({"year": 2000 + yr.astype(int),
                                        "month": pd.Series(mon, index=df.index).map(SLO_MONTH), "day": 1})
    df["strength_mg"] = df["strength"].str.extract(r"(\d+)")[0].astype(int)
    return df, norm


def robust_sigma(s: pd.Series) -> float:
    return 1.4826 * (s - s.median()).abs().median()


@lru_cache(maxsize=None)
def spec_table(data_dir: str) -> pd.DataFrame:
    df, _ = load_base(data_dir)
    spec = df.groupby("code").agg(**{"함량(mg/정)": ("strength_mg", "first")}, 배치수=("batch_yield", "size"),
                                  기준선=("batch_yield", "median"), sigma=("batch_yield", robust_sigma))
    spec["경고_하한"] = spec.기준선 - SIGMA_ALERT * spec.sigma
    spec["일탈_하한"] = spec.기준선 - SIGMA_ACTION * spec.sigma
    spec["상한"] = UPPER_LIMIT
    return spec


def judge(y: float, s: pd.Series, n: int) -> str:
    if y > UPPER_LIMIT:
        return "수율 일탈(상한 초과)"
    if n < MIN_BATCHES:
        return "판정 보류"
    if y < s.일탈_하한:
        return "수율 일탈(하한 미달)"
    if y < s.경고_하한:
        return "수율 이상(경고)"
    return "정상"


@lru_cache(maxsize=None)
def judged(data_dir: str) -> pd.DataFrame:
    df, _ = load_base(data_dir)
    spec = spec_table(data_dir)
    out = df[["code", "strength_mg", "batch_yield", "tbl_yield"]].copy()
    out["robust_z"] = (out.batch_yield - out.code.map(spec.기준선)) / out.code.map(spec.sigma)
    out["판정"] = [judge(r.batch_yield, spec.loc[r.code], spec.loc[r.code, "배치수"]) for r in out.itertuples()]
    return out


# ════════════════════════════════════════════════════════════════════
# 시계열 → 배치 지표
# ════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def load_product_ts(data_dir: str, code: int) -> pd.DataFrame:
    with zipfile.ZipFile(Path(data_dir) / "Process.zip") as z, z.open(f"Process/{code}.csv") as f:
        return pd.read_csv(f, sep=";")


def _segments(mask: np.ndarray):
    """True 연속 구간의 (시작, 끝(미포함)) 목록."""
    if len(mask) == 0:
        return []
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def batch_features(d: pd.DataFrame) -> pd.Series:
    d = d.reset_index(drop=True)
    speed = d.tbl_speed.to_numpy()
    fill = d.tbl_fill.to_numpy()
    stopped = speed == 0
    prod_idx = np.flatnonzero(d.produced.to_numpy() > 0)
    s = prod_idx[0] if len(prod_idx) else len(d)                 # 첫 양품 계수 = 셋업 종료
    p = stopped[s:]
    long_segs = [(a + s, b + s) for a, b in _segments(p) if b - a >= LONG_STOP]
    # 긴 정지 전후 충전 깊이 변화 (운전 중 샘플만, 정지 순간의 센서 튐 제외)
    n_fill_change = 0
    for a, b in long_segs:
        pre = fill[max(s, a - 30):a][speed[max(s, a - 30):a] > 0]
        post = fill[b:b + 30][speed[b:b + 30] > 0]
        if len(pre) >= 3 and len(post) >= 3 and abs(np.median(post) - np.median(pre)) >= 0.05:
            n_fill_change += 1
    early = d.iloc[s:s + EARLY_WIN]
    early_run = early.tbl_fill[early.tbl_speed > 0]
    m = re.search(r" (\d{1,2}):(\d{2})", d.timestamp.iat[0])
    return pd.Series({
        "campaign": d.campaign.iat[0],
        "start_hour": int(m.group(1)), "start_minute": int(m.group(2)),
        "n_samples": len(d),
        "setup_min": s * SAMPLE_SEC / 60,
        "setup_idle_min": stopped[:s].sum() * SAMPLE_SEC / 60,
        "prod_stop_min": p.sum() * SAMPLE_SEC / 60,
        "prod_long_stops": len(long_segs),
        "stop_fill_changes": n_fill_change,
        "early_fill_range": (early_run.max() - early_run.min()) if len(early_run) >= 30 else np.nan,
    })


def _ts_cands(s: str) -> dict:
    """'DDMMYYYY hh:mm' 문자열의 두 해석(원본 / 일·월 교환) 중 유효한 날짜."""
    m = re.match(r"(\d{2})(\d{2})(\d{4}) (\d{1,2}):(\d{2})", s)
    a, b, y, hh, mm = map(int, m.groups())
    out = {}
    for name, (day, mon) in {"원본": (a, b), "교환": (b, a)}.items():
        try:
            out[name] = pd.Timestamp(year=y, month=mon, day=day, hour=hh, minute=mm)
        except ValueError:
            pass
    return out


@lru_cache(maxsize=None)
def resolve_dates(data_dir: str, code: int) -> pd.DataFrame:
    """일·월 뒤바뀜 보정 (EDA 3.4).

    같은 제품 파일의 배치는 배치 번호 순서대로 생산되므로(시간 역행 없음), 각 배치의 시작·종료를
    {원본, 일·월 교환} 중에서 골라 '앞 배치 종료 ≤ 다음 배치 시작'을 지키면서 배치 사이 공백이
    가장 작은 해석의 조합을 동적계획법(Viterbi)으로 찾는다.
    """
    raw = load_product_ts(data_dir, code)
    g = raw.groupby("batch").timestamp
    first, last = g.first(), g.last()
    batches = sorted(first.index)
    states = []
    for b in batches:
        S, E = _ts_cands(first[b]), _ts_cands(last[b])
        st = [(ks, ke, S[ks], E[ke]) for ks in S for ke in E if pd.Timedelta(0) <= E[ke] - S[ks] <= pd.Timedelta(days=7)]
        states.append(st or [(ks, ks, S[ks], S[ks]) for ks in S])
    INF = 1e12
    cost = [[0.0 for _ in states[0]]]
    back = [[None for _ in states[0]]]
    for i in range(1, len(batches)):
        c_i, b_i = [], []
        for cur in states[i]:
            best, arg = INF * 10, 0
            for j, prv in enumerate(states[i - 1]):
                gap = (cur[2] - prv[3]).total_seconds() / 3600
                step = gap if gap >= 0 else INF
                if cost[-1][j] + step < best:
                    best, arg = cost[-1][j] + step, j
            c_i.append(best)
            b_i.append(arg)
        cost.append(c_i)
        back.append(b_i)
    k = int(np.argmin(cost[-1]))
    rows = {}
    for i in range(len(batches) - 1, -1, -1):
        ks, ke, s_, e_ = states[i][k]
        rows[batches[i]] = {"start_dt": s_, "end_dt": e_,
                            "date_method": "원본(일월연도)" if ks == "원본" else "일·월 교환",
                            "ambiguous": len(_ts_cands(first[batches[i]])) > 1}
        k = back[i][k] if back[i][k] is not None else 0
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


@lru_cache(maxsize=None)
def product_frame(data_dir: str, code: int) -> pd.DataFrame:
    df, norm = load_base(data_dir)
    raw = load_product_ts(data_dir, code)
    feat = raw.groupby("batch")[raw.columns].apply(batch_features)
    dp = df[df.code == code].join(feat).sort_index()
    dp["first_in_campaign"] = dp.groupby("campaign").cumcount().eq(0)
    dp["night_start"] = dp.start_hour.isin([22, 23, 0, 1, 2, 3, 4, 5])
    dp["is_weekend"] = dp.weekend.eq("yes")
    dp["season"] = dp.start_month.dt.month.map(SEASON)
    # 운전 설정값 이탈: 속도·주압축력 평균이 제품 중앙값에서 몇 % 벗어났나 (높아도 낮아도 나쁨 → 절댓값, 둘 중 큰 값)
    # σ̂ 배수 대신 %를 쓰는 이유: 설정값은 거의 고정이라 MAD가 0에 가까워 σ̂ 배수가 비현실적으로 커진다
    devs = [((dp[c] / dp[c].median()) - 1).abs() * 100 for c in ["tbl_speed_mean", "main_CompForce mean"]]
    dp["setting_dev"] = pd.concat(devs, axis=1).max(axis=1)
    f = norm[code]
    dp["loss_tab"] = dp["size"] * (100 - dp.batch_yield) / 100
    dp["w_setup"] = dp.startup_waste
    dp["w_prod"] = dp.total_waste * f - dp.startup_waste
    dp["w_after"] = dp["size"] * (dp.tbl_yield - dp.batch_yield).clip(lower=0) / 100   # 타정 이후 공정 손실
    dp["w_unrec"] = dp.loss_tab - dp.total_waste * f - dp.w_after
    dp["trace"] = (dp.total_waste * f) / dp.loss_tab
    return dp


# ════════════════════════════════════════════════════════════════════
# 가설 정의 (6M × 1~3개, 총 14개)
# ════════════════════════════════════════════════════════════════════
HYP = pd.DataFrame([
    # id, 6M, 원인 후보, 지표, 유형, 방향(+1: 클수록 나쁨, -1: 작을수록 나쁨), 지표 설명, 근본원인(RC), 상위 가설
    ("MAN-1", "인력", "야간 시간대 셋업 착수 (숙련 인력·지원 부족)", "night_start", "bin", +1, "배치 시작 22~06시", "RC-MAN-A", None),
    ("MAN-2", "인력", "주말 생산 (인력 축소 운영)", "is_weekend", "bin", +1, "주말 생산 여부", "RC-MAN-A", None),
    ("MAN-3", "인력", "운전 설정값 이탈 (설정 입력·레시피 선택 오류)", "setting_dev", "cont", +1, "속도·주압축력 설정 이탈(중앙값 대비 %)", "RC-MAN-B", None),
    ("MAC-1", "설비", "본생산 중 장시간 정지·재시작 반복", "prod_stop_min", "cont", +1, "본생산 중 정지 시간(분)", "RC-MAC", None),
    ("MAC-2", "설비", "주압축력 변동 (압축부 불안정)", "main_CompForce_sd", "cont", +1, "주압축력 표준편차", "RC-MAC", "MAC-1"),
    ("MAC-3", "설비", "정제 배출력 과다 (금형·펀치 마찰)", "ejection_max", "cont", +1, "배출력 최댓값", "RC-TOOL", None),
    ("MAT-1", "자재", "API 입도가 미세한 로트 (흐름성 저하)", "api_ps09", "cont", -1, "API 입도 D90(μm)", "RC-MAT", None),
    ("MAT-2", "자재", "API 수분 과다 (뭉침·부착)", "api_water", "cont", +1, "API 수분(%)", "RC-MAT", None),
    ("MET-1", "방법", "셋업 장기화", "setup_min", "cont", +1, "셋업 시간(분)", "RC-MET-A", None),
    ("MET-2", "방법", "충전 조건 반복 변경", "fom_change", "cont", +1, "충전 장치 속도 변경 횟수", "RC-MET-A", "MET-1"),
    ("MET-3", "방법", "셋업 직후 조건 재조정 (조건 미확정 상태로 본생산 시작)", "early_fill_range", "cont", +1, "셋업 후 60분 충전 깊이 변화폭", "RC-MET-B", None),
    ("MEA-1", "측정", "셋업 중 압축력 기반 무게 판정 신호(SREL) 과변동", "SREL_startup_mean", "cont", +1, "셋업 단계 SREL 평균", "RC-MEA", None),
    ("MEA-2", "측정", "정제 무게 편차 증가 (IPC)", "tbl_rsd_weight", "cont", +1, "정제 무게 RSD(%)", "RC-MEA", None),
    ("ENV-1", "환경", "생산 계절의 온·습도 변화 (대리 지표: 계절)", "same_season", "bin", +1, "대상 배치와 같은 계절 생산", "RC-ENV", None),
], columns=["id", "6M", "원인 후보", "지표", "유형", "방향", "지표 설명", "RC", "상위"]).set_index("id")

UP = {h: u for h, u in HYP["상위"].items() if isinstance(u, str)}   # 하위 가설 → 상위 가설
PROXY = {"ENV-1"}   # 대리 지표 가설: 지표가 가설의 조건(온·습도) 자체를 재지 않으므로 채택 대신 '보류(대리 지표)'까지만
SHORT = {"MAN-1": "야간 셋업 착수", "MAN-2": "주말 생산", "MAN-3": "운전 설정값 이탈",
         "MAC-1": "본생산 중 장시간 정지", "MAC-2": "주압축력 변동", "MAC-3": "배출력 과다",
         "MAT-1": "API 입도 미세", "MAT-2": "API 수분 과다",
         "MET-1": "셋업 장기화", "MET-2": "충전 조건 반복 변경", "MET-3": "셋업 직후 조건 재조정",
         "MEA-1": "셋업 중 SREL 과변동", "MEA-2": "정제 무게 편차", "ENV-1": "생산 계절 온·습도"}

RC = {
    "RC-MET-A": ("방법", "제품별로 검증된 초기 셋업 설정값(충전 깊이·압축력 레시피)이 정립되어 있지 않다.", "셋업"),
    "RC-MET-B": ("방법", "셋업 완료 판정 기준(연속 정제 규격 충족·압축력 안정 확인 등)이 정립되어 있지 않다.", "본생산"),
    "RC-MAC": ("설비", "재시작 후 공정 조건 안정화 확인 절차와 압축부 상태 점검·경보 기준이 없다.", "본생산"),
    "RC-TOOL": ("설비", "금형·펀치 상태 점검·교체 기준(사용 횟수·배출력 경보)이 없다.", "본생산"),
    "RC-MAN-A": ("인력", "야간·주말 셋업과 이상 대응에 숙련자 배치·지원 요청(에스컬레이션) 기준이 없다.", "셋업"),
    "RC-MAN-B": ("인력", "운전 설정값 이중 확인과 레시피 선택 검증 절차가 없다.", "전체"),
    "RC-MAT": ("자재", "API 입도·수분 규격에 타정 공정 적합성(흐름성)이 반영되어 있지 않다.", "전체"),
    "RC-MEA": ("측정", "셋업 단계의 무게 측정 체계(SREL 판정 한계·센서 확인·IPC 주기) 확인 기준이 없다.", "셋업"),
    "RC-ENV": ("환경", "계절별 작업장 온·습도 관리 기준(경보·공조 조정 절차)이 없다.", "전체"),
}
M_ORDER = [("인력", "Man"), ("설비", "Machine"), ("자재", "Material"),
           ("방법", "Method"), ("측정", "Measurement"), ("환경", "Environment")]
TAG_COLOR = {"데이터": C_NORMAL, "데이터+추정": "#8a6d1f", "추정": C_GRAY}


# ════════════════════════════════════════════════════════════════════
# 통계 도구
# ════════════════════════════════════════════════════════════════════
def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m, order = len(p), np.argsort(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def partial_spearman(data: pd.DataFrame, x: str, y: str, covars: list):
    """순위 변환 후 공변량을 회귀로 제거한 잔차끼리의 상관 (양측 p)."""
    r = data[[x, y] + covars].astype(float).dropna().rank()
    Z = np.column_stack([np.ones(len(r))] + [r[c].to_numpy() for c in covars])
    resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]   # noqa: E731
    rho = np.corrcoef(resid(r[x].to_numpy()), resid(r[y].to_numpy()))[0, 1]
    dof = len(r) - 2 - len(covars)
    t = rho * np.sqrt(dof / (1 - rho ** 2))
    return rho, 2 * stats.t.sf(abs(t), dof)


def form_html(title: str, doc_no: str, rows: list) -> str:
    th = 'style="background:#f1f0ec;text-align:left;padding:6px 8px;white-space:nowrap;vertical-align:top"'
    td = 'style="text-align:left;padding:6px 8px;vertical-align:top"'
    sec = 'style="background:#2a3b4c;color:white;text-align:left;padding:6px 8px;font-weight:bold"'
    h = [f'<h3 style="margin-bottom:2px">{title}</h3><p style="margin-top:0;color:#52514e">문서번호: {doc_no}</p>',
         '<table style="border-collapse:collapse;width:100%;font-size:13px" border="1">',
         f"<tr><th {th}>구분</th><th {th}>상세 기재 내용</th><th {th}>구분</th><th {th}>상세 기재 내용</th></tr>"]
    for r in rows:
        if r[0] == "§":
            h.append(f'<tr><td colspan="4" {sec}>{r[1]}</td></tr>')
        elif len(r) == 2:
            h.append(f'<tr><th {th}>{r[0]}</th><td colspan="3" {td}>{r[1]}</td></tr>')
        else:
            h.append(f"<tr><th {th}>{r[0]}</th><td {td}>{r[1]}</td><th {th}>{r[2]}</th><td {td}>{r[3]}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# RCA 본체
# ════════════════════════════════════════════════════════════════════
class RCA:
    def __init__(self, batch: int, data_dir: str | Path | None = None):
        self.data_dir = str(data_dir or find_data_dir())
        self.df, self.norm = load_base(self.data_dir)
        self.batch = int(batch)
        self.code = int(self.df.loc[self.batch, "code"])
        self.spec = spec_table(self.data_dir)
        self.sp = self.spec.loc[self.code]
        self.raw = load_product_ts(self.data_dir, self.code)
        self.dp = product_frame(self.data_dir, self.code).copy()
        self.T = self.dp.loc[self.batch]
        self.dp["same_season"] = self.dp.season.eq(self.T.season)
        self.T = self.dp.loc[self.batch]
        self.ref = self.dp.drop(index=self.batch)
        self.판정 = judge(self.T.batch_yield, self.sp, self.sp.배치수)
        self._dates()
        self._numbers()
        self._observe()
        self._five_why()
        self._stage12()
        self._stage3()

    # ── 날짜 ────────────────────────────────────────────────────────
    def _dates(self):
        d = resolve_dates(self.data_dir, self.code).loc[self.batch]
        self.start_dt, self.end_dt, self.date_method = d.start_dt, d.end_dt, d.date_method
        self.date_ambiguous = bool(d.ambiguous)
        self.setup_end_dt = self.start_dt + pd.Timedelta(minutes=float(self.T.setup_min))

    # ── 숫자 모음 ───────────────────────────────────────────────────
    def _numbers(self):
        T, ref = self.T, self.ref
        f = self.norm[self.code]
        loss = T.loss_tab
        self.E = E = {
            "yield": T.batch_yield, "tbl_yield": T.tbl_yield, "ref_mean": ref.batch_yield.mean(),
            "ref_med": ref.batch_yield.median(), "n_ref": len(ref),
            "limit": self.sp.일탈_하한, "alert": self.sp.경고_하한,
            "loss": loss, "good": T["size"] * T.batch_yield / 100, "size": T["size"],
            "w_setup": T.w_setup, "w_prod": T.w_prod, "w_after": T.w_after, "w_unrec": T.w_unrec,
            "recorded": T.total_waste * f, "trace": T.trace,
            "setup_min": T.setup_min, "setup_idle": T.setup_idle_min,
            "pstop": T.prod_stop_min, "nstops": T.prod_long_stops, "campaign": int(T.campaign),
            "first": bool(T.first_in_campaign), "pos": int((self.dp.campaign == T.campaign).loc[:self.batch].sum()),
        }
        for k, col in [("setup_w_med", "w_setup"), ("prod_w_med", "w_prod"), ("unrec_med", "w_unrec"),
                       ("setup_med", "setup_min"), ("pstop_med", "prod_stop_min"), ("nstops_med", "prod_long_stops")]:
            E[k] = ref[col].median()
        E["gap_limit"] = E["yield"] - (E["limit"] if E["yield"] < E["limit"] else E["alert"])
        E["gap_mean"] = E["yield"] - E["ref_mean"]
        for k in ["w_setup", "w_prod", "w_after", "w_unrec"]:
            E[k + "_share"] = E[k] / loss if loss > 0 else np.nan
        tw = ((self.dp.tbl_min_weight + self.dp.tbl_max_weight) / 2).median()
        E["tablet_mg"] = tw
        E["mass_in"], E["mass_out"] = E["size"] * tw / 1e6, E["good"] * tw / 1e6

    # ── 단계 ④: 관찰 근거 ──────────────────────────────────────────
    def extremity(self, var, direction):
        x, v = self.ref[var].dropna(), self.T[var]
        if pd.isna(v) or len(x) == 0:
            return np.nan
        k = (x >= v).sum() if direction > 0 else (x <= v).sum()
        return (k + 1) / (len(x) + 1)

    def _observe(self):
        rows = []
        for hid, h in HYP.iterrows():
            v = self.T[h.지표]
            if h.유형 == "cont":
                ext = self.extremity(h.지표, h.방향)
                obs = "데이터 없음" if pd.isna(ext) else ("이상 관찰" if ext < ALPHA else "해당 없음")
                rows.append({"790 값": v, "참조 중앙값": self.ref[h.지표].median(), "극단 백분위": ext, "관찰 판정": obs})
            else:
                rows.append({"790 값": bool(v), "참조 중앙값": self.ref[h.지표].mean(), "극단 백분위": np.nan,
                             "관찰 판정": "해당" if bool(v) else "해당 없음"})
        self.hyp = HYP.join(pd.DataFrame(rows, index=HYP.index).rename(columns={"790 값": "대상 값"}))
        self.observed = list(self.hyp.index[self.hyp["관찰 판정"].isin(["이상 관찰", "해당"])])
        self.screened = [h for h in self.hyp.index if h not in self.observed]

    def pct(self, hid):
        e = self.hyp.loc[hid, "극단 백분위"]
        return f"{'상위' if HYP.loc[hid, '방향'] > 0 else '하위'} {e:.1%}"

    def fmt_val(self, hid, v=None):
        h = HYP.loc[hid]
        v = self.hyp.loc[hid, "대상 값"] if v is None else v
        if h.유형 == "bin":
            return "예" if v else "아니오"
        return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"

    # ── 단계 ⑤: 가설별 5 Why ───────────────────────────────────────
    def _five_why(self):
        T, E, V = self.T, self.E, lambda c: self.T[c]
        M = lambda c: self.ref[c].median()  # noqa: E731
        Q1 = f"왜 배치 {self.batch}의 수율이 {E['yield']:.1f}%로 떨어졌나?"
        obs = set(self.observed)
        join_note = lambda h: f" **→ {h}와 합류**" if h in obs else ""  # noqa: E731
        if E["first"]:
            camp = (f"캠페인 {E['campaign']}의 첫 배치(제품 교체 직후)였다. 직전 제품 설정에서 출발했을 가능성이 있다.", "데이터+추정")
        else:
            camp = (f"캠페인 {E['campaign']}의 {E['pos']}번째 배치로 제품 교체 직후가 아니다. 설정을 바꾼 이유는 데이터로 알 수 없다.", "추정")
        stop_why = (
            (f"1분 이상 정지 {E['nstops']:.0f}회 중 {T.stop_fill_changes:.0f}회는 정지 전후로 충전 깊이 설정이 바뀌었다. "
             "조건을 재조정하려고 세웠을 가능성이 있지만, 정지 이유 자체는 기록에 없다.", "데이터+추정")
            if T.stop_fill_changes > 0 else
            ("정지 전후로 충전 깊이 설정 변화는 없었다. 정지 이유(자재 공급·설비 고장·대기 등)는 데이터로 알 수 없다.", "추정"))
        lab = lambda hid, u="": f"{self.fmt_val(hid)}{u} (제품 중앙값 {self.fmt_val(hid, self.hyp.loc[hid, '참조 중앙값'])}{u}, {self.pct(hid)})"  # noqa: E731
        fw = {
            "MAN-1": [
                (Q1, f"배치를 {self.start_dt:%H:%M}(야간)에 시작했다. 셋업 {E['setup_min']:.0f}분 중 {E['setup_idle']:.0f}분은 기계가 멈춰 있었다.", "데이터"),
                ("왜 야간 셋업에서 멈춘 시간이 길었나?", "설정 변경 뒤 확인·판단을 기다리는 시간이 길었을 수 있다.", "추정"),
                ("왜 확인·판단이 늦었나?", "야간조에 숙련 셋업 인력·QC 지원이 없었을 수 있다.", "추정"),
                ("왜 지원 인력이 없었나?", "야간 셋업을 배정할 때 인력 구성을 함께 정하지 않았을 수 있다.", "추정"),
                ("왜 함께 정하지 않았나?", RC["RC-MAN-A"][1], "추정")],
            "MAN-2": [
                (Q1, "주말에 생산했다.", "데이터"),
                ("왜 주말 생산이 문제가 되나?", "축소 인력으로 셋업·이상 대응이 늦었을 수 있다.", "추정"),
                ("왜 대응이 늦었나?", "숙련자·정비 인력이 호출 대기였을 수 있다.", "추정"),
                ("왜 호출 대기였나?", "주말 근무 편성 시 셋업·정비 대응 역량을 고려하지 않았을 수 있다.", "추정"),
                ("왜 고려하지 않았나?", RC["RC-MAN-A"][1], "추정")],
            "MAN-3": [
                (Q1, f"운전 설정이 제품 기준에서 크게 벗어났다 — 타정 속도 평균 {V('tbl_speed_mean'):.0f} (중앙값 {M('tbl_speed_mean'):.0f}), "
                     f"주압축력 평균 {V('main_CompForce mean'):.1f} (중앙값 {M('main_CompForce mean'):.1f}), 중앙값 대비 최대 {V('setting_dev'):.0f}% 이탈.", "데이터"),
                ("왜 설정이 벗어났나?", "다른 제품의 설정이 적용됐거나 설정값이 잘못 입력됐을 수 있다.", "추정"),
                ("왜 잘못된 설정이 그대로 운전됐나?", "설정값을 입력한 뒤 제품 기준과 대조하지 않았을 수 있다.", "추정"),
                ("왜 대조하지 않았나?", "운전 절차에 설정값 이중 확인(작업자·확인자) 항목이 없을 수 있다.", "추정"),
                ("왜 이중 확인 항목이 없나?", RC["RC-MAN-B"][1], "추정")],
            "MAC-1": [
                (Q1, f"본생산 중 정지가 {lab('MAC-1', '분')}, 1분 이상 정지가 {E['nstops']:.0f}회였다.", "데이터"),
                ("왜 정지가 길고 잦았나?", *stop_why),
                ("왜 조건 재조정이 필요했나?", f"주압축력 변동이 {lab('MAC-2')}였다. 압축부 불안정이 원인인지, 잦은 재조정의 결과인지는 구분할 수 없다.{join_note('MAC-2')}", "데이터+추정"),
                ("왜 불안정한 상태로 생산을 재개했나?", "재시작 후 조건 안정화와 압축부 상태를 확인하지 않고 양품 계수를 재개했을 수 있다.", "추정"),
                ("왜 확인하지 않았나?", RC["RC-MAC"][1], "추정")],
            "MAC-2": [
                (Q1, f"주압축력 표준편차가 {lab('MAC-2')}였다.", "데이터"),
                ("왜 주압축력이 흔들렸나?", "압축부(펀치·압축 롤러·압축력 센서) 상태가 불안정했을 수 있다.", "추정"),
                ("왜 불안정한 상태가 생산 중에야 드러났나?", "압축력 변동을 실시간으로 알리는 경보가 없어 기준을 벗어난 뒤에야 대응했을 수 있다.", "추정"),
                ("왜 사후에야 대응했나?", "제품 교체·재시작 시 압축부 상태 확인이 필수 항목이 아닐 수 있다.", "추정"),
                ("왜 필수 항목이 아닌가?", RC["RC-MAC"][1], "추정")],
            "MAC-3": [
                (Q1, f"배출력 최댓값이 {lab('MAC-3')}였다.", "데이터"),
                ("왜 배출력이 컸나?", "금형·펀치 표면 마모나 분말 부착으로 정제가 빠질 때 마찰이 컸을 수 있다.", "추정"),
                ("왜 마모·부착이 생겼나?", "금형·펀치 점검·세척 주기가 사용량에 비해 길었을 수 있다.", "추정"),
                ("왜 생산 전에 발견되지 않았나?", "배출력 상승을 알리는 경보·점검 항목이 없을 수 있다.", "추정"),
                ("왜 없나?", RC["RC-TOOL"][1], "추정")],
            "MAT-1": [
                (Q1, f"API 입도 D90이 {lab('MAT-1', ' μm')}로 미세한 로트가 투입됐다.", "데이터"),
                ("왜 미세한 입도가 문제가 되나?", "혼합 분말의 흐름성·밀도가 평소와 달라 충전 무게를 맞추기 어려웠을 수 있다.", "추정"),
                ("왜 그런 로트가 투입됐나?", "현행 입도 규격 범위 안이라 수입 시험에서 적합으로 판정됐을 수 있다.", "추정"),
                ("왜 규격 안인데 공정에 문제가 되나?", "현행 규격이 품질(용출) 기준으로 정해져 타정 흐름성은 고려하지 않았을 수 있다.", "추정"),
                ("왜 흐름성을 고려하지 않았나?", RC["RC-MAT"][1], "추정")],
            "MAT-2": [
                (Q1, f"API 수분이 {lab('MAT-2', '%')}였다.", "데이터"),
                ("왜 수분이 문제가 되나?", "분말 뭉침·부착으로 흐름성이 떨어졌을 수 있다.", "추정"),
                ("왜 수분이 높은 로트가 투입됐나?", "현행 수분 규격 범위 안이라 적합으로 판정됐을 수 있다.", "추정"),
                ("왜 규격 안인데 문제가 되나?", "현행 규격이 타정 흐름성 영향을 고려하지 않았을 수 있다.", "추정"),
                ("왜 고려하지 않았나?", RC["RC-MAT"][1], "추정")],
            "MET-1": [
                (Q1, f"셋업이 {lab('MET-1', '분')} 걸렸다.", "데이터"),
                ("왜 셋업이 길어졌나?", f"셋업 중 설정이 자주 바뀌었다 — 충전 장치 속도 변경 {V('fom_change'):.0f}회 (중앙값 {M('fom_change'):.0f}회), "
                                      f"셋업 중 충전 깊이 최대 변화폭 {V('Startup_tbl_fill_maxDifference'):.2f} (중앙값 {M('Startup_tbl_fill_maxDifference'):.2f}). "
                                      f"이 변경이 셋업을 늘린 원인인지, 길어진 셋업 동안 쌓인 결과인지는 구분할 수 없다.{join_note('MET-2')}", "데이터+추정"),
                ("왜 설정을 자주 바꿨나?", *camp),
                ("왜 제품 조건으로 바로 맞추지 못했나?", "검증된 초기 설정값이 셋업 문서에 없었을 수 있다.", "추정"),
                ("왜 문서에 없나?", RC["RC-MET-A"][1], "추정")],
            "MET-2": [
                (Q1, f"충전 장치 속도 변경이 {lab('MET-2', '회')}였다. 변경 주체(작업자·자동 제어)는 데이터로 구분할 수 없다.", "데이터"),
                ("왜 반복 변경됐나?", "목표 무게·경도를 맞출 시작 조건 없이 결과를 보며 조정했을 수 있다.", "추정"),
                ("왜 시작 조건이 없었나?", *camp),
                ("왜 제품 조건으로 바로 맞추지 못했나?", "검증된 초기 설정값이 셋업 문서에 없었을 수 있다.", "추정"),
                ("왜 문서에 없나?", RC["RC-MET-A"][1], "추정")],
            "MET-3": [
                (Q1, f"셋업 종료 후 60분 안에 충전 깊이가 {lab('MET-3')} 범위에서 다시 바뀌었다.", "데이터"),
                ("왜 본생산 직후 다시 조정했나?", "셋업 종료 시점에 조건이 아직 안정되지 않았다. 조정이 필요했던 것은 데이터, '미안정'이 이유라는 것은 해석이다.", "데이터+추정"),
                ("왜 안정 전에 본생산으로 넘어갔나?", "객관적 완료 조건(연속 정제 규격 충족 등) 없이 작업자 판단으로 양품 계수를 시작했을 수 있다.", "추정"),
                ("왜 작업자 판단에 맡겼나?", "셋업 완료 확인(IPC 결과·확인자) 기록 절차가 없을 수 있다.", "추정"),
                ("왜 절차가 없나?", RC["RC-MET-B"][1], "추정")],
            "MEA-1": [
                (Q1, f"셋업 중 압축력 기반 무게 판정 신호(SREL) 평균이 {lab('MEA-1')}였다.", "데이터"),
                ("왜 SREL이 흔들렸나?", "압축력 설정을 바꾼 직후 신호가 안정되기 전에 판정했을 수 있다.", "추정"),
                ("왜 안정 전에 판정했나?", "셋업 단계에도 정상 운전과 같은 판정 한계를 썼을 수 있다.", "추정"),
                ("왜 같은 한계를 썼나?", "셋업 단계의 판정 한계·센서 확인 항목이 정해져 있지 않을 수 있다.", "추정"),
                ("왜 정해져 있지 않나?", RC["RC-MEA"][1], "추정")],
            "MEA-2": [
                (Q1, f"IPC 정제 무게 RSD가 {lab('MEA-2', '%')}였다.", "데이터"),
                ("왜 무게 편차가 컸나?", f"충전 깊이 설정이 여러 번 바뀌었다(셋업 중 최대 변화폭 {V('Startup_tbl_fill_maxDifference'):.2f}). "
                                       "변경이 편차의 원인인지 결과인지는 구분할 수 없다.", "데이터+추정"),
                ("왜 여러 번 바꿨나?", "IPC 무게 결과가 늦게 반영되어 과보정·재보정이 반복됐을 수 있다.", "추정"),
                ("왜 늦게 반영됐나?", "셋업 단계의 IPC 측정 주기가 정상 운전과 같을 수 있다.", "추정"),
                ("왜 따로 정하지 않았나?", RC["RC-MEA"][1], "추정")],
            "ENV-1": [
                (Q1, f"{T.season}({self.start_dt:%m}월)에 생산했다. 작업장 온·습도는 수집되지 않는다.", "데이터"),
                ("왜 계절이 문제가 되나?", "작업장 온·습도가 평소와 달라 분말 흐름성이 변했을 수 있다.", "추정"),
                ("왜 온·습도가 달랐나?", "외기 변화에 맞춰 공조 설정이 조정되지 않았을 수 있다.", "추정"),
                ("왜 조정되지 않았나?", "온·습도 이탈 경보와 조정 절차가 없을 수 있다.", "추정"),
                ("왜 없나?", RC["RC-ENV"][1], "추정")],
        }
        loss_at = {"MAN-1": "셋업", "MAN-2": "셋업·본생산", "MAN-3": "셋업·본생산 전체", "MAC-1": "본생산 — 재시작마다 폐기 누적",
                   "MAC-2": "본생산 — 무게·경도 흔들림 → 정지·배출", "MAC-3": "본생산 — 정제 결함 배출",
                   "MAT-1": "셋업·본생산 — 충전 무게 맞추기 어려움", "MAT-2": "셋업·본생산 — 흐름성 저하",
                   "MET-1": "셋업 — 시험 타정 정제 폐기", "MET-2": "셋업 — 조정마다 시험 타정 반복",
                   "MET-3": "본생산 초반 — 조건 재조정 중 배출", "MEA-1": "셋업 — 판정 한계 밖 정제 자동 배출",
                   "MEA-2": "셋업·본생산 — 무게 기준 이탈 배출", "ENV-1": "셋업·본생산 — 흐름성 변화"}
        self.five_why = {h: {"chain": fw[h], "loss_at": loss_at[h]} for h in self.observed}

    # ── 단계 ⑥: 검정 ───────────────────────────────────────────────
    def _stage12(self):
        rows = []
        for hid in self.observed:
            h = HYP.loc[hid]
            r = self.ref[[h.지표, "batch_yield"]].dropna()
            v = self.T[h.지표]
            if h.유형 == "cont":
                s1p = self.extremity(h.지표, h.방향)
                s1 = s1p < ALPHA
                res = stats.spearmanr(r[h.지표], r.batch_yield)
                eff_name, eff, p = "스피어만 ρ", res.statistic, res.pvalue
                dir_ok = eff < 0 if h.방향 > 0 else eff > 0
                med = r[h.지표].median()
                thr = med + (v - med) / 2
                n_near = int((r[h.지표] >= thr).sum() if h.방향 > 0 else (r[h.지표] <= thr).sum())
                n = len(r)
            else:
                s1p, s1 = np.nan, bool(v)
                yes, no = r.loc[r[h.지표].astype(bool), "batch_yield"], r.loc[~r[h.지표].astype(bool), "batch_yield"]
                if len(yes) and len(no):
                    p = stats.mannwhitneyu(yes, no, alternative="two-sided").pvalue
                    eff = cliffs_delta(yes, no)
                else:
                    p, eff = 1.0, 0.0
                eff_name, dir_ok, n_near, n = "클리프 δ", eff < 0, len(yes), f"{len(yes)} vs {len(no)}"
            rows.append({"가설": hid, "RC": h.RC, "6M": h["6M"], "검증 지표": h["지표 설명"],
                         "① 극단 p": s1p, "① 통과": s1, "② 효과": f"{eff_name} = {eff:+.2f}", "효과값": eff,
                         "방향 일치": dir_ok, "② n": n, "비슷한 수준 참조 수": n_near, "② p (양측)": p})
        T = pd.DataFrame(rows).set_index("가설") if rows else pd.DataFrame(
            columns=["RC", "6M", "검증 지표", "① 극단 p", "① 통과", "② 효과", "효과값", "방향 일치", "② n",
                     "비슷한 수준 참조 수", "② p (양측)"])
        if len(T):
            T["② p (Holm)"] = holm(T["② p (양측)"])
            sig = T["② p (Holm)"] < ALPHA
            proxy = T.index.isin(list(PROXY))
            T["판정"] = np.select(
                [T["① 통과"] & sig & T["방향 일치"] & proxy,
                 T["① 통과"] & sig & T["방향 일치"],
                 T["① 통과"] & sig & ~T["방향 일치"],
                 T["① 통과"] & (T["비슷한 수준 참조 수"] < MIN_NEAR)],
                ["보류(대리 지표)", "채택", "기각(반대 방향)", "보류(참조 부족)"], "기각")
        else:
            T["② p (Holm)"], T["판정"] = [], []
        self.test = T

    def _stage3(self):
        T = self.test
        adopted = list(T.index[T.판정 == "채택"]) if len(T) else []
        rc_cand = [rc for rc in RC if any(HYP.loc[h, "RC"] == rc for h in adopted)]
        rep = {}
        for rc in rc_cand:
            hs = [h for h in adopted if HYP.loc[h, "RC"] == rc]
            roots = [h for h in hs if UP.get(h) not in hs] or hs
            rep[rc] = min(roots, key=lambda h: T.loc[h, "② p (Holm)"])
        checks = []
        for h in adopted:
            up = UP.get(h)
            if up and up in self.observed:
                r = stats.spearmanr(self.ref[HYP.loc[h, "지표"]], self.ref[HYP.loc[up, "지표"]], nan_policy="omit")
                checks.append((HYP.loc[h, "RC"], "(a) 합류 고리", f"{h} {SHORT[h]} ↑ ↔ {up} {SHORT[up]} ↑",
                               "스피어만 ρ (양측)", r.statistic, r.pvalue, "연관 있음" if (r.pvalue < ALPHA and r.statistic > 0) else "연관 확인 안 됨"))
        for i, a in enumerate(rc_cand):
            for b in rc_cand[i + 1:]:
                xa, xb = HYP.loc[rep[a], "지표"], HYP.loc[rep[b], "지표"]
                r = stats.spearmanr(self.ref[xa].astype(float), self.ref[xb].astype(float), nan_policy="omit")
                checks.append((f"{a}·{b}", "(b) 독립성", f"{rep[a]} ↔ {rep[b]}", "스피어만 ρ (양측)", r.statistic, r.pvalue,
                               "독립" if abs(r.statistic) < 0.2 else "연관"))
        indep = {}
        for rc in rc_cand:
            x = HYP.loc[rep[rc], "지표"]
            others = [HYP.loc[rep[o], "지표"] for o in rc_cand if o != rc]
            rho, p = partial_spearman(self.ref, x, "batch_yield", others)
            ok = (rho < 0 if HYP.loc[rep[rc], "방향"] > 0 else rho > 0) and p < ALPHA
            indep[rc] = (rho, p, ok)
            checks.append((rc, "(b) 독립 효과", f"{rep[rc]} → 수율 | " + (", ".join(others) + " 통제" if others else "통제 없음"),
                           "편 스피어만 ρ (양측)", rho, p, "독립 효과 있음" if ok else "독립 효과 확인 안 됨"))
        if "RC-MET-A" in rc_cand:
            fy, fn = self.ref.loc[self.ref.first_in_campaign, "setup_min"], self.ref.loc[~self.ref.first_in_campaign, "setup_min"]
            p = stats.mannwhitneyu(fy, fn, alternative="two-sided").pvalue
            dlt = cliffs_delta(fy, fn)
            checks.append(("RC-MET-A", "(c) 5 Why 고리", f"캠페인 첫 배치 → 셋업 시간 (중앙값 {fy.median():.0f}분 vs {fn.median():.0f}분)"
                           + ("" if self.E["first"] else " — 대상 배치는 첫 배치 아님"),
                           "클리프 δ (맨-휘트니 양측)", dlt, p, "연관 있음" if (p < ALPHA and dlt > 0) else "연관 확인 안 됨"))
        self.chk = pd.DataFrame(checks, columns=["RC", "구분", "확인 내용", "방법", "통계량", "p", "결과"])
        rows, hold = [], {}
        for rc, (m_ko, text, phase) in RC.items():
            hs = [h for h in self.observed if HYP.loc[h, "RC"] == rc]
            if not hs:
                v = "해당 없음"
            elif rc in indep and indep[rc][2]:
                v = "채택"
            elif rc in indep:
                v = "기각(독립 효과 없음)"
            elif any(T.loc[h, "판정"].startswith("보류") for h in hs):
                v = "보류(현장 확인)"
                hold[rc] = ("대리 지표(계절)만으로는 판단할 수 없음 → 작업장 온·습도 기록 확인"
                            if any(T.loc[h, "판정"] == "보류(대리 지표)" for h in hs)
                            else f"참조 배치 중 비슷한 수준의 사례가 {MIN_NEAR}개 미만이라 통계로 판단할 수 없음 → 현장 기록 확인")
            else:
                v = "기각"
            rows.append({"RC": rc, "관점": m_ko, "근본원인 후보": text,
                         "연결 가설 (판정)": ", ".join(f"{h}({T.loc[h, '판정']})" for h in hs) or "—",
                         "③ 독립 효과": (f"편 ρ = {indep[rc][0]:+.2f}, p = {indep[rc][1]:.2g}" if rc in indep else "—"),
                         "RC 판정": v})
        V = pd.DataFrame(rows).set_index("RC")
        adopted_rc = sorted([rc for rc in V.index if V.loc[rc, "RC 판정"] == "채택"], key=lambda rc: -abs(indep[rc][0]))
        self.R_TAG = {rc: f"R{i}" for i, rc in enumerate(adopted_rc, 1)}
        V["보고서 번호"] = V.index.map(lambda rc: self.R_TAG.get(rc, "—"))
        self.rc_verdict, self.rep, self.indep = V, rep, indep
        self.held = list(V.index[V["RC 판정"] == "보류(현장 확인)"])
        self.held_reason = hold

    # ════════════════════════════════════════════════════════════════
    # 표시 함수 — 노트북 각 단계에서 호출
    # ════════════════════════════════════════════════════════════════
    @property
    def kind(self):
        return "수율 일탈" if "일탈" in self.판정 else "수율 이상"

    def show_overview(self):
        E = self.E
        warn = ""
        if E["trace"] < 0.5:
            warn = (f"\n\n> ⚠ **데이터 RCA 범위 주의** — 손실의 {E['trace']:.0%}만 타정기 폐기 기록으로 추적됩니다. "
                    "이 노트북의 결론은 기록된 손실에 대한 것이며, 나머지는 현장 기록(청소·수작업 폐기·집진·샘플링)으로 확인해야 합니다.")
        display(Markdown(
            f"| 항목 | 내용 |\n|---|---|\n| 대상 | 제품 {self.code} (함량 {self.T.strength_mg} mg) · 배치 {self.batch} |\n"
            f"| 판정 | **{self.판정}** — 수율 {E['yield']:.2f}% (경고 하한 {E['alert']:.2f}%, 일탈 하한 {E['limit']:.2f}%) |\n"
            f"| 생산 일시 | {self.start_dt:%Y-%m-%d %H:%M} ~ {self.end_dt:%Y-%m-%d %H:%M} (날짜 해석: {self.date_method}) |\n"
            f"| 비교 기준 | 제품 {self.code}의 다른 배치 {E['n_ref']}개 (평균 수율 {E['ref_mean']:.2f}%) |\n"
            f"| 손실 추적 가능 비율 | {E['trace']:.0%} (타정기 폐기 기록 ÷ 손실) |" + warn))

    def show_spec(self):
        j = judged(self.data_dir)
        sub = j[j.code == self.code]
        s = self.sp
        display(Markdown(
            f"**제품 {self.code} 수율 규격** — 기준선(중앙값) {s.기준선:.2f}%, σ̂ = {s.sigma:.2f}%p → "
            f"경고 하한 {s.경고_하한:.2f}%, 일탈 하한 {s.일탈_하한:.2f}%, 상한 {UPPER_LIMIT:.1f}%  \n"
            f"제품 {self.code} 배치 {len(sub)}개 중: " + ", ".join(f"{k} {v}개" for k, v in sub.판정.value_counts().items())))
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.scatter(range(len(self.dp)), self.dp.batch_yield, s=12, color=C_NORMAL, alpha=0.6, linewidths=0, label=f"제품 {self.code} 배치")
        flag = sub.판정.ne("정상").reindex(self.dp.index)
        ax.scatter(np.flatnonzero(flag.to_numpy()), self.dp.batch_yield[flag], s=26, facecolors="none", edgecolors=C_LOW, label="수율 이상·일탈 배치")
        pos = self.dp.index.get_loc(self.batch)
        ax.scatter(pos, self.E["yield"], s=80, color=C_LOW, zorder=3, label=f"배치 {self.batch}")
        ax.annotate(f"{self.batch}", (pos, self.E["yield"]), xytext=(6, -12), textcoords="offset points", color=C_LOW, fontweight="bold")
        for y, lab_, ls in [(s.기준선, f"기준선 {s.기준선:.2f}%", "-"), (s.경고_하한, f"경고 하한 {s.경고_하한:.2f}%", "--"),
                            (s.일탈_하한, f"일탈 하한 {s.일탈_하한:.2f}%", "-")]:
            ax.axhline(y, color=C_LOW if "일탈" in lab_ else C_TEXT2, lw=1, ls=ls)
            ax.text(len(self.dp) + 1, y + 0.04, lab_, va="bottom", fontsize=8, color=C_TEXT2)
        ax.set(title=f"제품 {self.code} 수율 관리도 (배치 번호 순)", xlabel="배치 순서", ylabel="batch_yield (%)",
               xlim=(-3, len(self.dp) + 30), ylim=(max(self.dp.batch_yield.min() - 0.5, 80), 100.5))
        ax.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        plt.show()

    def _normal_batch(self):
        r = self.ref
        pool = r[(~r.first_in_campaign) & (r.setup_min <= r.setup_min.median()) & (r.prod_stop_min <= r.prod_stop_min.median())]
        pool = pool if len(pool) else r
        return int((pool.batch_yield - r.batch_yield.median()).abs().idxmin())

    def show_trace(self):
        nb = self._normal_batch()
        fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True)
        for b, color, name in [(self.batch, C_LOW, f"배치 {self.batch} (대상)"), (nb, C_NORMAL, f"배치 {nb} (대표 정상)")]:
            t = self.raw[self.raw.batch == b].reset_index(drop=True)
            h = t.index * SAMPLE_SEC / 3600
            for ax, col in zip(axes, ["tbl_speed", "tbl_fill", "main_comp", "waste"]):
                y = t[col].cummax() if col == "waste" else t[col]
                ax.plot(h, y, color=color, lw=1.1, label=name)
                ax.axvline(self.dp.loc[b, "setup_min"] / 60, color=color, lw=1, ls="--")
        run = self.raw[self.raw.tbl_speed > 0]
        axes[1].set_ylim(run.tbl_fill.quantile(0.002) - 0.2, run.tbl_fill.quantile(0.998) + 0.2)
        axes[2].set_ylim(0, run.main_comp.quantile(0.999) * 1.15)
        for ax, lab_ in zip(axes, ["타정 속도", "충전 깊이", "주압축력", "누적 폐기 정제 수"]):
            ax.set_ylabel(lab_)
        axes[3].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axes[0].set_title("10초 시계열 — 대상 배치 vs 대표 정상 배치 (점선 = 셋업 종료 = 첫 양품 계수)")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("배치 시작 후 경과 시간 (시간)")
        plt.tight_layout()
        plt.show()
        E = self.E
        display(Markdown(
            f"> **관찰** — 셋업 {E['setup_min']:.0f}분 (제품 중앙값 {E['setup_med']:.0f}분, 그중 정지 {E['setup_idle']:.0f}분), "
            f"본생산 중 정지 {E['pstop']:.0f}분 (중앙값 {E['pstop_med']:.0f}분), 1분 이상 정지 {E['nstops']:.0f}회 (중앙값 {E['nstops_med']:.0f}회). "
            f"대표 정상 배치는 {nb}입니다(캠페인 첫 배치가 아니고 셋업·정지 시간이 중앙값 이하인 배치 중 수율이 중앙값에 가장 가까운 배치)."))

    def show_loss(self):
        E = self.E
        parts = [("셋업 폐기", "w_setup"), ("본생산 폐기", "w_prod"), ("타정 이후 공정 손실", "w_after"), ("미기록 손실", "w_unrec")]
        tab = pd.DataFrame({
            "대상 배치 (정)": [E[k] for _, k in parts],
            "손실 중 비중": [E[k + "_share"] for _, k in parts],
            "제품 중앙값 (정)": [self.ref[k].median() for _, k in parts],
        }, index=[n for n, _ in parts])
        tab["중앙값 대비"] = [(a - b) for a, b in zip(tab["대상 배치 (정)"], tab["제품 중앙값 (정)"])]
        tab.loc["손실 합계"] = [E["loss"], 1.0, self.ref.loss_tab.median(), E["loss"] - self.ref.loss_tab.median()]
        display(tab.style.format({"대상 배치 (정)": "{:,.0f}", "손실 중 비중": "{:.0%}", "제품 중앙값 (정)": "{:,.0f}",
                                  "중앙값 대비": "{:+,.0f}"}).set_caption(
            f"손실 분해 — 이론 {E['size']:,.0f}정, 양품 {E['good']:,.0f}정 (질량 환산 추정: 투입 {E['mass_in']:,.1f} kg, 양품 {E['mass_out']:,.1f} kg)"))
        ex = tab.drop(index="손실 합계")["중앙값 대비"]
        top = ex.idxmax()
        display(Markdown(
            f"> **해석** — 평소(제품 중앙값)보다 가장 많이 늘어난 손실은 **{top}**(+{ex.max():,.0f}정)입니다. "
            + " ".join(f"{n} {v:+,.0f}정." for n, v in ex.items() if n != top)
            + (" 미기록 손실은 타정기 카운터 밖 손실이라 데이터로 원인을 좇을 수 없습니다." if ex["미기록 손실"] > 0 else "")))

    def show_5w1h(self):
        E, T = self.E, self.T
        where = "타정 공정" + (f" + 타정 이후 공정(타정 수율 {E['tbl_yield']:.2f}% → 배치 수율 {E['yield']:.2f}%)" if E["w_after"] > 0 else
                              " (배치 수율 = 타정 수율 수준, 이후 공정 손실 없음)")
        rows = [
            ("What (무엇이)", f"제품 {self.code}(함량 {T.strength_mg} mg) 배치 {self.batch}의 배치 수율이 **{E['yield']:.2f}%**로 "
                             f"{'일탈 하한 ' + format(E['limit'], '.2f') if self.kind == '수율 일탈' else '경고 하한 ' + format(E['alert'], '.2f')}%를 "
                             f"{abs(E['gap_limit']):.2f}%p 밑돌았다 (참조 평균 대비 {E['gap_mean']:+.2f}%p)."),
            ("When (언제)", f"{self.start_dt:%Y-%m-%d %H:%M} ~ {self.end_dt:%Y-%m-%d %H:%M}"
                            f"{' (날짜는 일·월 교환 해석)' if self.date_method != '원본(일월연도)' else ''}. "
                            f"캠페인 {E['campaign']}의 {'첫 배치(제품 교체 직후)' if E['first'] else str(E['pos']) + '번째 배치'}. "
                            f"셋업 {self.start_dt:%H:%M}~{self.setup_end_dt:%H:%M}."),
            ("Where (어디서)", f"{where}. 손실 위치: 셋업 {E['w_setup_share']:.0%} · 본생산 {E['w_prod_share']:.0%} · "
                              f"이후 공정 {E['w_after_share']:.0%} · 미기록 {E['w_unrec_share']:.0%}."),
            ("Who (누가)", f"발견·보고: 공정 엔지니어 [가상]. 관련: {self.start_dt:%H}시에 셋업을 시작한 작업조 (데이터에 인력 정보 없음)."),
            ("Why (왜)", f"**미확정 — RCA로 규명.** 관찰: 셋업 {E['setup_min']:.0f}분(중앙값 {E['setup_med']:.0f}분), "
                         f"본생산 중 정지 {E['pstop']:.0f}분(중앙값 {E['pstop_med']:.0f}분)."),
            ("How (어떻게·얼마나)", f"이론 {E['size']:,.0f}정 중 양품 {E['good']:,.0f}정, 손실 **{E['loss']:,.0f}정** = 셋업 폐기 {E['w_setup']:,.0f} "
                                  f"+ 본생산 폐기 {E['w_prod']:,.0f} + 이후 공정 {E['w_after']:,.0f} + 미기록 {E['w_unrec']:,.0f}."),
        ]
        display(Markdown("| 항목 | 내용 |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in rows)))

    def _report_quality(self):
        out = []
        for c, name in [("dissolution_av", "용출 평균(%)"), ("dissolution_min", "용출 최소(%)"), ("impurities_total", "총 불순물(%)")]:
            lo, hi, v = self.ref[c].min(), self.ref[c].max(), self.T[c]
            out.append(f"{name} {v:.2f} ({'범위 내' if lo <= v <= hi else '범위 밖'}: {lo:.2f}~{hi:.2f})")
        return " · ".join(out)

    def show_incident_report(self):
        E = self.E
        inc = f"INC-{self.start_dt:%Y-%m%d}-{self.batch}"
        self.inc_no = inc
        kind_box = ("<b>[X] 수율 이상</b> &nbsp; [ ] 수율 일탈" if self.kind == "수율 이상" else "[ ] 수율 이상 &nbsp; <b>[X] 수율 일탈</b>")
        lim = E["limit"] if self.kind == "수율 일탈" else E["alert"]
        display(HTML(form_html("수율 이상 발생 보고서", f"{inc} <i>(예시 번호)</i>", [
            ("발생 일시", f"{self.start_dt:%Y.%m.%d. %H:%M} ~ {self.end_dt:%Y.%m.%d. %H:%M}", "최초 발견 일시",
             f"{self.end_dt + pd.Timedelta(minutes=15):%Y.%m.%d. %H:%M} [가상]"),
            ("보고 일시", f"{self.end_dt + pd.Timedelta(minutes=45):%Y.%m.%d. %H:%M} [가상]", "보고자", "공정 엔지니어 (가명) [가상]"),
            ("대상 제품", f"제품 {self.code} (함량 {self.T.strength_mg} mg 정제, 배치 크기 {E['size']:,.0f}정)", "배치 번호",
             f"{self.batch} (캠페인 {E['campaign']} {'첫 배치' if E['first'] else str(E['pos']) + '번째'})"),
            ("발생 공정", "타정" + (" (+ 타정 이후 공정)" if E["w_after"] > 0 else ""), "설비명 (번호)", "타정기 (설비 번호 데이터 미제공)"),
            ("발생 유형", kind_box, "관할 부서", "생산기술팀 (공정 엔지니어)"),
            ("§", "2. 발생 및 수율 상세 내용"),
            ("대상 수율", f"<b>{E['yield']:.2f}%</b>", "비교 배치 / 평균 수율", f"{E['n_ref']}개 배치 / 평균 {E['ref_mean']:.2f}%"),
            ("수율 차이", f"<b>{E['gap_mean']:+.2f}%p</b>", "현장 점검 결과",
             f"타정기 기록 점검: 셋업 {E['setup_min']:.0f}분(정지 {E['setup_idle']:.0f}분), 본생산 중 1분 이상 정지 {E['nstops']:.0f}회"),
            ("발생 내용 상세", f"이론 {E['size']:,.0f}정 중 양품 {E['good']:,.0f}정. 셋업 폐기 {E['w_setup']:,.0f}정, 본생산 폐기 {E['w_prod']:,.0f}정, "
                             f"타정 이후 공정 손실 {E['w_after']:,.0f}정, 미기록 손실 {E['w_unrec']:,.0f}정. 원인 미확정."),
            ("§", "3. 수율 산정 기준 및 판정"),
            ("투입 질량", f"{E['mass_in']:,.2f} kg (추정: {E['size']:,.0f}정 × {E['tablet_mg']:.0f} mg)", "회수 질량", f"{E['mass_out']:,.2f} kg (추정)"),
            ("공정 경계", "타정기 투입(최종 혼합 분말)부터 최종 양품 정제까지. 배치 수율 = 양품 ÷ 배치 크기(이론 정제 수) × 100."),
            ("적용 허용범위", f"{lim:.2f}% ~ {UPPER_LIMIT:.1f}% ({'일탈' if self.kind == '수율 일탈' else '경고'} 하한: 제품 {self.code} 중앙값 − "
                             f"{SIGMA_ACTION if self.kind == '수율 일탈' else SIGMA_ALERT}σ̂)",
             "수율 허용범위 판정", "[ ] 허용범위 내 &nbsp; <b>[X] 하한 미달</b> &nbsp; [ ] 상한 초과 &nbsp; [ ] 판정 대기"),
            ("§", "4. 제품 품질 영향 평가"),
            ("초기 원인 추정 및 공정 영향", f"6M 관점의 원인 후보를 RCA로 조사 예정이며 확정 원인은 없다. 출하 품질 시험 결과: {self._report_quality()}. 최종 품질 영향은 QA 평가 중."),
            ("§", "5. 초기 조치 및 담당자·실시 현황"),
            ("초기 조치 내용", "[가상] 배치 식별·격리, 후속 공정 이송 보류, 타정기 운전 기록·IPC 기록 보존.", "담당자", "공정 엔지니어 (가명) [가상]"),
            ("실시 일시", f"{self.end_dt + pd.Timedelta(minutes=20):%Y.%m.%d. %H:%M} [가상]", "완료 여부", "<b>[X] 완료</b> &nbsp; [ ] 진행 중"),
            ("제품 처분 상태", "<b>[X] 출하 보류 및 격리</b> &nbsp; [ ] 폐기 &nbsp; [ ] 재가공 &nbsp; [ ] 기타 — 최종 처분은 QA 평가 후 결정"),
        ])))

    def show_hypotheses(self):
        h = self.hyp.copy()
        out = pd.DataFrame({
            "6M": h["6M"], "원인 후보": h["원인 후보"], "지표": h["지표 설명"],
            "대상 값": [self.fmt_val(i) for i in h.index],
            "참조": [(f"중앙값 {self.fmt_val(i, h.loc[i, '참조 중앙값'])}" if HYP.loc[i, "유형"] == "cont"
                      else f"참조 중 {h.loc[i, '참조 중앙값']:.0%}") for i in h.index],
            "극단 백분위": h["극단 백분위"], "관찰 판정": h["관찰 판정"]})
        display(out.style.format({"극단 백분위": "{:.1%}"}, na_rep="—")
                .apply(lambda r: ["color: #9a9893" if r["관찰 판정"] in ("해당 없음", "데이터 없음") else "" for _ in r], axis=1)
                .set_caption(f"6M 가설 {len(h)}개와 배치 {self.batch}의 관찰 근거 (회색 = 관찰되지 않음 → Why 1에서 종결)"))
        display(Markdown(f"> **관찰된 가설 {len(self.observed)}개**: " + ", ".join(f"{i} {SHORT[i]}" for i in self.observed)
                         + f"  \n> **종결 {len(self.screened)}개**: " + ", ".join(f"{i} {SHORT[i]}" for i in self.screened)))

    def show_five_why(self):
        td = "style='padding:5px 8px;vertical-align:top'"
        for rc in RC:
            hs = [h for h in self.observed if HYP.loc[h, "RC"] == rc]
            if not hs:
                continue
            display(Markdown(f"#### {rc} ({RC[rc][0]}) 로 이어지는 경로"))
            for hid in hs:
                w = self.five_why[hid]
                up = UP.get(hid)
                rows = "".join(
                    f"<tr><td {td}><b>Why {i}</b></td><td {td}>{q}</td><td {td}>{a.replace('**', '')}</td>"
                    f"<td style='padding:5px 8px;white-space:nowrap;color:{TAG_COLOR[tag]}'>[{tag}]</td></tr>"
                    for i, (q, a, tag) in enumerate(w["chain"], 1))
                join = f" &nbsp;·&nbsp; <b>{up}의 원인 쪽(하위 가설)</b>" if up and up in self.observed else ""
                display(HTML(
                    f"<h4 style='margin:14px 0 4px'>{hid} · {HYP.loc[hid, '6M']} — {HYP.loc[hid, '원인 후보']}</h4>"
                    f"<p style='margin:0 0 4px;color:#52514e'>작용 위치: {w['loss_at']}{join}</p>"
                    "<table border='1' style='border-collapse:collapse;font-size:13px;width:100%'>"
                    "<tr style='background:#f1f0ec'><th>단계</th><th>질문</th><th>답</th><th>근거</th></tr>"
                    f"{rows}<tr style='background:#fde7dd'><td {td}><b>근본원인 후보</b></td>"
                    f"<td colspan='3' {td}><b>{rc}</b> · {RC[rc][1]}</td></tr></table>"))

    def show_rc_table(self):
        rows = []
        for rc, (m_ko, text, _) in RC.items():
            hs = [h for h in self.observed if HYP.loc[h, "RC"] == rc]
            if not hs:
                continue
            ab = {"데이터": "데", "데이터+추정": "데+", "추정": "추"}
            tags = [f"{h}: " + "·".join(ab[t] for *_, t in self.five_why[h]["chain"]) for h in hs]
            rows.append({"RC": rc, "관점": m_ko, "근본원인 후보 (Why 5)": text, "연결 가설": ", ".join(hs),
                         "가설 간 관계": "; ".join(f"{h} → {UP[h]}" for h in hs if UP.get(h) in hs) or
                         ("병렬" if len(hs) > 1 else "단일 경로"),
                         "Why 1~5 근거 (데=데이터, 데+=데이터+추정, 추=추정)": " / ".join(tags)})
        display(pd.DataFrame(rows).set_index("RC") if rows else Markdown("관찰된 가설이 없어 근본원인 후보가 없습니다."))

    def show_tests(self):
        if not len(self.test):
            display(Markdown("관찰된 가설이 없어 검정할 것이 없습니다."))
            return
        t = self.test.drop(columns=["효과값"])
        display(t.style.format({"① 극단 p": "{:.3f}", "② p (양측)": "{:.2g}", "② p (Holm)": "{:.2g}"}, na_rep="해당(예)")
                .apply(lambda r: ["background-color: #fde7dd; font-weight: bold" if r.판정 == "채택" else
                                  ("background-color: #fff4d6" if r.판정.startswith("보류") else "color: #52514e") for _ in r], axis=1)
                .set_caption(f"가설 {len(t)}개 2단계 검정 (참조 {self.E['n_ref']}개, α = {ALPHA}, Holm 보정 {len(t)}개, 양측 + 방향 일치)"))
        n = len(self.observed)
        cols = 3
        rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(15, 4.1 * rows_), squeeze=False)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for ax, hid in zip(axes.flat, self.observed):
            h, t_ = HYP.loc[hid], self.test.loc[hid]
            color = C_LOW if t_.판정 == "채택" else ("#b08a1e" if t_.판정.startswith("보류") else C_GRAY)
            if h.유형 == "cont":
                ax.scatter(self.ref[h.지표], self.ref.batch_yield, s=12, alpha=0.55, color=C_NORMAL, linewidths=0, label="참조 배치")
                ax.scatter(self.T[h.지표], self.E["yield"], s=80, color=C_LOW, zorder=3, label=f"배치 {self.batch}")
                x = self.dp[h.지표].dropna()
                if len(x) and x.min() >= 0 and x.max() > 20 and x.max() / max(x.median(), 1e-9) > 8:
                    ax.set_xscale("symlog")
                    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                g = [self.ref.loc[~self.ref[h.지표].astype(bool), "batch_yield"], self.ref.loc[self.ref[h.지표].astype(bool), "batch_yield"]]
                ax.boxplot([x if len(x) else [np.nan] for x in g], positions=[0, 1], widths=0.5,
                           medianprops=dict(color=C_TEXT2), flierprops=dict(markersize=3, markeredgecolor=C_GRAY))
                ax.scatter(1, self.E["yield"], s=80, color=C_LOW, zorder=3, label=f"배치 {self.batch}")
                ax.set_xticks([0, 1], [f"아니오 (n={len(g[0])})", f"예 (n={len(g[1])})"])
            ax.set_xlabel(h["지표 설명"])
            ax.axhline(self.sp.일탈_하한, color=C_LOW, lw=0.8, ls="--")
            ax.set_title(f"{hid} ({h.RC}) → {t_.판정}\n{t_['② 효과']}, Holm p = {t_['② p (Holm)']:.2g}", color=color, fontsize=11)
            ax.set_ylabel("batch_yield (%)")
        axes.flat[0].legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        plt.show()
        display(Markdown(self._interp_tests()))

    def _interp_tests(self):
        t = self.test
        lines = ["> **해석**", ">"]
        for verdict, msg in [("채택", "대상 배치 말고 다른 배치들에서도 이 조건이 수율 저하와 함께 나타납니다"),
                             ("기각", "대상 배치에서는 관찰됐지만, 비슷한 수준의 다른 배치들은 수율이 평소와 다르지 않았습니다 → 원인이라기보다 동반 신호"),
                             ("기각(반대 방향)", "평소 데이터에서는 오히려 가설과 반대 방향의 관계가 나타납니다 → 가설 메커니즘 기각, 반대 메커니즘 검토 필요"),
                             ("보류(참조 부족)", f"참조 배치 중 대상 배치와 비슷한 수준의 조건이 {MIN_NEAR}개 미만이라 통계로 판단할 수 없습니다 → 현장 기록으로 확인"),
                             ("보류(대리 지표)", "평소에도 수율과 연관되지만 지표가 가설의 조건 자체(온·습도)를 재지 않는 대리 지표라 근본원인으로 채택하지 않습니다 → 작업장 온·습도 기록으로 확인")]:
            hs = list(t.index[t.판정 == verdict])
            if hs:
                lines.append(f"> - **{verdict}** ({len(hs)}개: {', '.join(f'{h} {SHORT[h]}' for h in hs)}) — {msg}.")
        weak = t[(t.판정 == "기각") & (t["② p (양측)"] < ALPHA) & t["방향 일치"]]
        if len(weak):
            lines.append(f"> - 보정 전에는 유의했지만 Holm 보정 후 기각된 가설: {', '.join(weak.index)} — 증거가 약한 것이지 관계가 없다는 뜻은 아니므로 추가 확인 항목으로 남깁니다.")
        return "\n".join(lines)

    def show_stage3(self):
        if not len(self.chk):
            display(Markdown("> 채택된 가설이 없어 근본원인 확인(③)을 하지 않습니다."))
        else:
            display(self.chk.style.format({"통계량": "{:+.2f}", "p": "{:.2g}"}).hide(axis="index")
                    .set_caption("근본원인 확인 — 합류 고리 · 독립성 · 5 Why 데이터 고리"))
        V = self.rc_verdict[self.rc_verdict["RC 판정"] != "해당 없음"]
        display(V.style.apply(lambda r: ["background-color: #fde7dd; font-weight: bold" if r["RC 판정"] == "채택" else
                                         ("background-color: #fff4d6" if r["RC 판정"].startswith("보류") else "color: #52514e") for _ in r], axis=1)
                .set_caption("근본원인 후보 최종 판정 (관찰된 가설이 있는 RC만)"))
        msg = []
        if self.R_TAG:
            msg.append("> **채택 근본원인**: " + " / ".join(f"**{tag} [{RC[rc][0]}]** {RC[rc][1]}" for rc, tag in self.R_TAG.items()))
        else:
            msg.append("> **채택 근본원인 없음** — 데이터로 지지되는 근본원인을 찾지 못했습니다. 관찰된 조건들은 다른 배치에서 수율 저하와 함께 나타나지 않았습니다.")
        if self.held:
            msg.append(">\n> **보류(현장 확인)**: " + " / ".join(f"{rc} ({RC[rc][1]}) — {self.held_reason[rc]}" for rc in self.held))
        dropped = list(V.index[V["RC 판정"] == "기각(독립 효과 없음)"])
        if dropped:
            msg.append(">\n> **가설은 채택됐지만 근본원인으로는 기각**: " + ", ".join(dropped)
                       + " — 다른 채택 근본원인을 통제하면 수율 효과가 남지 않아, 같은 현상을 중복으로 센 것으로 봅니다.")
        display(Markdown("\n".join(msg)))

    def _whatif(self):
        phases = {RC[rc][2] for rc in self.R_TAG}
        E, ref = self.E, self.ref
        rows = [("실제", E["good"])]
        g = E["good"]
        if "셋업" in phases and E["w_setup"] > ref.w_setup.median():
            g += E["w_setup"] - ref.w_setup.median()
            rows.append(("셋업 관련 근본원인 해소: 셋업 폐기 → 중앙값", g))
        if "본생산" in phases and E["w_prod"] > ref.w_prod.median():
            g += E["w_prod"] - ref.w_prod.median()
            rows.append(("본생산 관련 근본원인 해소: 본생산 폐기 → 중앙값", g))
        w = pd.DataFrame(rows, columns=["시나리오", "양품 (정)"])
        w["수율 (%)"] = w["양품 (정)"] / E["size"] * 100
        w["판정"] = [judge(y, self.sp, self.sp.배치수) for y in w["수율 (%)"]]
        return w

    def show_whatif(self):
        if not self.R_TAG:
            display(Markdown("> 채택된 근본원인이 없어 해소 효과를 추정하지 않습니다."))
            return
        w = self._whatif()
        if len(w) == 1:
            display(Markdown("> 채택된 근본원인이 작용하는 손실 구간(셋업·본생산 폐기)이 평소보다 많지 않아, 폐기 기록으로 해소 효과를 추정할 수 없습니다."))
            return
        display(w.style.format({"양품 (정)": "{:,.0f}", "수율 (%)": "{:.2f}"}).hide(axis="index")
                .set_caption(f"근본원인 해소 시 추정 수율 (단순 산술, 경고 하한 {self.E['alert']:.2f}%, 일탈 하한 {self.E['limit']:.2f}%)"))

    def draw_fishbone(self, after: bool):
        status = {}
        for hid in HYP.index:
            if hid in self.screened:
                status[hid] = ("H", False, False, "(해당 없음)")
            elif not after:
                status[hid] = ("H", False, False, "")
            else:
                rc, v = HYP.loc[hid, "RC"], self.test.loc[hid, "판정"]
                up = UP.get(hid)
                if v == "채택" and rc in self.R_TAG:
                    status[hid] = (self.R_TAG[rc], True, True, f"({up}의 하위 원인)" if up and up in self.observed else "(채택)")
                elif v == "채택":
                    status[hid] = ("H", True, False, "(가설 채택·독립 효과 없음)")
                elif v.startswith("보류"):
                    status[hid] = ("H", False, False, "(보류: 현장 확인)")
                else:
                    status[hid] = ("H", False, False, "(검정 기각)")
        fig, ax = plt.subplots(figsize=(15, 7.4))
        ax.set_xlim(-3.4, 16.4)
        ax.set_ylim(-4.3, 4.3)
        ax.axis("off")
        ax.annotate("", xy=(12.6, 0), xytext=(-0.6, 0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=C_TEXT2))
        ax.text(14.5, 0, f"{self.kind}\n제품 {self.code} · 배치 {self.batch}\n{self.E['yield']:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.6", fc=C_LOW, ec="none"))
        for k, (m_ko, m_en) in enumerate(M_ORDER):
            top_ = k < 3
            x1 = [3.0, 7.2, 11.4][k % 3]
            x0, y0 = x1 - 2.4, (3.3 if top_ else -3.3)
            ax.plot([x0, x1], [y0, 0], color=C_TEXT2, lw=1.6)
            ax.text(x0, y0 + (0.38 if top_ else -0.38), f"{m_ko} ({m_en})", ha="center", va="center", fontsize=10.5,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f1f0ec", ec="#c3c2b7"))
            for j, hid in enumerate(HYP.index[HYP["6M"] == m_ko]):
                tag, solid, strong, note = status[hid]
                t = [0.25, 0.52, 0.79][j]
                px, py = x0 + t * (x1 - x0), y0 + t * (0 - y0)
                color = C_LOW if strong else (C_TEXT2 if solid else C_GRAY)
                ax.plot([px - 0.45, px], [py, py], color=color, lw=1.8 if strong else 1.2, ls="-" if solid else (0, (3, 2)))
                ax.text(px - 0.52, py, f"[{tag}] {SHORT[hid]}" + (f"\n{note}" if note else ""), ha="right", va="center",
                        fontsize=9, color=color, fontweight="bold" if strong else "normal", linespacing=1.15)
        ax.set_title(f"6M 특성요인도 — {'검정 후' if after else '초기 가설'}", loc="left", fontsize=12, fontweight="bold")
        ax.text(-3.3, -4.2, "[H] 초기 가설   [R] 채택된 근본원인   실선 = 검증 근거 확보   점선 = 미검증 연결", fontsize=8.5, color=C_TEXT2)
        plt.show()

    def show_rca_report(self):
        E = self.E
        T = self.test
        inc = getattr(self, "inc_no", f"INC-{self.start_dt:%Y-%m%d}-{self.batch}")

        def hyp_ev(h):
            t = T.loc[h]
            s1 = f"① p = {t['① 극단 p']:.3f}" if HYP.loc[h, "유형"] == "cont" else "① 해당"
            return f"{h} {SHORT[h]} ({s1}, ② {t['② 효과']}, Holm p = {t['② p (Holm)']:.2g})"

        def chk_txt(rc):
            c = self.chk[self.chk.RC.str.contains(rc, regex=False)]
            return " ".join(f"{r['구분']}: {r['확인 내용']} → {r['결과']} ({r['방법'].split(' ')[0]} {r['통계량']:+.2f}, p = {r['p']:.2g})."
                            for _, r in c.iterrows())

        rows = [
            ("§", "1. 조사 기본정보"),
            ("담당부서", "생산기술팀", "담당자", "공정 엔지니어 (가명) [가상]"),
            ("대상 제품", f"제품 {self.code} (함량 {self.T.strength_mg} mg 정제)", "배치 번호", str(self.batch)),
            ("대상 공정", "타정" + (" (+ 타정 이후 공정)" if E["w_after"] > 0 else ""), "대상 설비", "타정기 (설비 번호 데이터 미제공)"),
            ("조사 시스템", "<b>■ MES</b>(타정기 시계열·공정 요약) &nbsp; <b>■ LIMS</b>(원료·IPC·출하 시험) &nbsp; □ QMS &nbsp; <b>■ 설비 이력</b> &nbsp; □ 기타"),
            ("조사 기간 및 비교 범위", f"제품 {self.code} 전체 {len(self.dp)}개 배치({self.dp.start_month.min():%Y-%m} ~ {self.dp.start_month.max():%Y-%m}). "
                                     f"비교 기준 = 배치 {self.batch}를 제외한 {E['n_ref']}개 배치."),
            ("§", "2. 관찰된 문제 및 조사 결과"),
            ("관찰된 문제", f"{self.판정} — 셋업 {E['setup_min']:.0f}분, 본생산 중 정지 {E['pstop']:.0f}분(1분 이상 {E['nstops']:.0f}회)"),
            ("대상 배치 수율", f"{E['yield']:.2f}% (경고 하한 {E['alert']:.2f}%, 일탈 하한 {E['limit']:.2f}%)", "비교 배치 평균 수율",
             f"{E['n_ref']}개 배치 / 평균 {E['ref_mean']:.2f}%"),
            ("손실 내역 및 물질수지", f"이론 {E['size']:,.0f}정 = 양품 {E['good']:,.0f} + 셋업 폐기 {E['w_setup']:,.0f}({E['w_setup_share']:.0%}) "
                                    f"+ 본생산 폐기 {E['w_prod']:,.0f}({E['w_prod_share']:.0%}) + 이후 공정 {E['w_after']:,.0f}({E['w_after_share']:.0%}) "
                                    f"+ 미기록 {E['w_unrec']:,.0f}({E['w_unrec_share']:.0%}) 정. 손실 추적 가능 비율 {E['trace']:.0%}."),
            ("§", "3. 특성요인도 및 6M 조사"),
            ("6M 특성요인도", f"6M × 가설 {len(HYP)}개 (관찰 {len(self.observed)}개에 가설별 5 Why 적용) — 노트북의 검정 후 특성요인도 첨부"),
            ("범주 및 표시", "[H] 초기 가설 / [R] 채택된 근본원인 / 실선 = 검증 근거 확보 / 점선 = 미검증 연결"),
            ("기타 가설 검토 결과", "<br>".join(
                [f"• {h} {SHORT[h]}: {T.loc[h, '판정']} ({T.loc[h, '② 효과']}, Holm p = {T.loc[h, '② p (Holm)']:.2g})"
                 for h in T.index if T.loc[h, "판정"] != "채택"]
                + [f"• 관찰되지 않아 종결: {', '.join(f'{h}({SHORT[h]})' for h in self.screened)}"])),
            ("§", "4. 채택된 근본원인 및 근거"),
        ]
        if self.R_TAG:
            for rc, tag in self.R_TAG.items():
                hs = [h for h in self.observed if HYP.loc[h, "RC"] == rc and T.loc[h, "판정"] == "채택"]
                rows.append((f"{tag} 근본원인", f"<b>[{RC[rc][0]}] {RC[rc][1]}</b>"))
                rows.append((f"{tag} 채택 이유 및 검증 근거", "채택 가설: " + " / ".join(hyp_ev(h) for h in hs) + ". " + chk_txt(rc)))
                rows.append((f"{tag} 5Why 경로", " ← ".join(a.replace("**", "").split(" — ")[0].split(". ")[0]
                                                         for _, a, _ in self.five_why[self.rep[rc]]["chain"])))
        else:
            rows.append(("근본원인", "데이터로 지지되는 근본원인 없음"))
        lim = ["Why 2~5의 [추정] 단계(관리상 원인 포함)는 데이터로 확인 불가 → SOP·셋업 기록·설정 변경 이력(변경 주체)·정비 이력 확인 필요."]
        if E["w_unrec"] > self.ref.w_unrec.median():
            lim.append(f"미기록 손실 {E['w_unrec']:,.0f}정(제품 중앙값 {self.ref.w_unrec.median():,.0f}정)은 카운터 밖 손실로 원인 미확정 → 청소·수작업 폐기·집진·샘플링 기록 확인.")
        if self.held:
            lim.append("보류(현장 확인): " + " ".join(f"{rc} — {self.held_reason[rc]}." for rc in self.held))
        weak = T[(T.판정 == "기각") & (T["② p (양측)"] < ALPHA) & T["방향 일치"]] if len(T) else T
        if len(weak):
            lim.append("보정 후 기각됐지만 방향이 맞고 보정 전 유의: " + ", ".join(f"{h}({SHORT[h]})" for h in weak.index) + " → 추가 확인.")
        lim.append("관찰 데이터 기반 연관성 검정이므로 인과는 CAPA 시행 후 효과성 평가로 최종 확인.")
        rows.append(("조사 한계 및 추가 확인", " ".join(f"{'①②③④⑤⑥'[i]} {x}" for i, x in enumerate(lim))))
        display(HTML(form_html("근본원인 조사 보고서 (RCA)", f"RCA-{self.start_dt:%Y-%m%d}-{self.batch} <i>(예시 번호, 발생보고서 {inc} 연계)</i>", rows)))

    def summary(self) -> dict:
        return {"제품": self.code, "배치": self.batch, "수율(%)": round(self.E["yield"], 2), "판정": self.판정,
                "추적 가능": f"{self.E['trace']:.0%}", "관찰 가설": len(self.observed),
                "채택 가설": ", ".join(self.test.index[self.test.판정 == "채택"]) if len(self.test) else "",
                "채택 근본원인": " / ".join(f"{tag} {rc}" for rc, tag in self.R_TAG.items()) or "없음",
                "보류": ", ".join(self.held) or "—"}


def _conclusion(self: RCA):
    E = self.E
    lines = [f"**결론** — 배치 {self.batch}(제품 {self.code})는 수율 {E['yield']:.2f}%로 **{self.판정}**입니다."]
    if self.R_TAG:
        lines.append("데이터로 지지되는 근본원인은 " + ", ".join(
            f"**{tag} [{RC[rc][0]}]** {RC[rc][1]}" for rc, tag in self.R_TAG.items()) + "")
        w = self._whatif()
        if len(w) > 1:
            last = w.iloc[-1]
            lines.append(f"채택 근본원인이 작용한 폐기를 제품 중앙값 수준으로 되돌리면 수율은 약 {last['수율 (%)']:.2f}%({last['판정']})로 추정됩니다.")
    else:
        lines.append("데이터로 지지되는 근본원인은 찾지 못했습니다. 관찰된 조건들은 다른 배치에서 수율 저하와 함께 나타나지 않았습니다.")
    if self.held:
        lines.append("**현장 확인이 필요한 후보**: " + " ".join(f"{rc}({RC[rc][0]}) — {self.held_reason[rc]}." for rc in self.held))
    if E["trace"] < 0.5:
        lines.append(f"손실의 {E['trace']:.0%}만 타정기 기록으로 추적되어, 결론은 기록된 손실에 한정됩니다.")
    lines.append("근본원인의 Why 2~5 중 [추정] 단계는 SOP·셋업 기록·설정 변경 이력·정비 이력으로 확인해야 합니다.")
    display(Markdown("> " + "\n>\n> ".join(lines)))


RCA.show_conclusion = _conclusion
