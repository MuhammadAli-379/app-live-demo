
import json
import textwrap
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN TOKENS
# =========================================================

COLORS = {
    "burgundy": "#4A0717",
    "burgundy_light": "#780B27",
    "burgundy_dark": "#32040F",
    "gold": "#E6C866",
    "text": "#271C20",
    "muted": "#6B6265",
    "border": "#E8E2DF",
    "background": "#F7F5F3",
    "green": "#2F9455",
    "green_soft": "#EAF6EE",
    "orange": "#D58A2D",
    "orange_soft": "#FFF3E4",
    "red": "#B32A45",
    "red_soft": "#FCECEF",
    "blue": "#3867A8",
    "blue_soft": "#EDF3FB",
    "white": "#FFFFFF",
}

CSS_VARS = "".join(
    f"--{k.replace('_', '-')}: {v};"
    for k, v in COLORS.items()
)


CSS_RULES = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    color-scheme: light;
}

html, body, [class*="st-"], .stApp {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.main .block-container {
    max-width: 1380px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

/* =====================================================
   GLOBAL TYPOGRAPHY
   ===================================================== */

h1, h2, h3, h4, h5 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--text) !important;
}

[data-testid="stMarkdownContainer"] > p,
[data-testid="stMarkdownContainer"] > ul,
[data-testid="stMarkdownContainer"] > ol,
[data-testid="stMarkdownContainer"] > ul li,
[data-testid="stMarkdownContainer"] > ol li,
[data-testid="stMarkdownContainer"] > h1,
[data-testid="stMarkdownContainer"] > h2,
[data-testid="stMarkdownContainer"] > h3,
[data-testid="stMarkdownContainer"] > h4 {
    color: var(--text) !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--text) !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: var(--burgundy) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {
    color: var(--muted) !important;
}

/* =====================================================
   METRICS
   ===================================================== */

[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--border);
    padding: .75rem .9rem;
    border-radius: 12px;
}

/* =====================================================
   EXPANDERS / ALERTS
   ===================================================== */

[data-testid="stExpander"] details {
    background: #fff;
    border-color: var(--border);
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--text) !important;
}

[data-testid="stAlert"] {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p {
    color: var(--text) !important;
}

/* =====================================================
   TABS
   ===================================================== */

.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p {
    color: var(--muted) !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p {
    color: var(--burgundy) !important;
}

/* =====================================================
   INPUTS
   ===================================================== */

[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="select"] > div {
    background: #fff !important;
}

[data-baseweb="input"] input,
[data-baseweb="select"] *,
[data-testid="stNumberInput"] button {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background: #fff !important;
    color: var(--text) !important;
}

/* =====================================================
   SIDEBAR
   ===================================================== */

[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] :is(
    h1, h2, h3, h4, p, label, li, small
) {
    color: var(--text) !important;
}

[data-testid="stSidebar"]
[data-testid="stCaptionContainer"] * {
    color: var(--muted) !important;
}

/* =====================================================
   CONTAINERS
   ===================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border-radius: 14px;
    border-color: var(--border);
}

/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button,
[data-testid="stDownloadButton"] button {
    width: 100%;
    min-height: 2.6rem;
    border-radius: 10px;
    font-weight: 600;
    background: #fff;
    border: 1px solid var(--border);
}

.stButton > button p,
[data-testid="stDownloadButton"] button p {
    color: var(--text) !important;
}

.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    min-height: 3.05rem;
    border: none;
    background:
        linear-gradient(
            120deg,
            var(--burgundy),
            var(--burgundy-light)
        ) !important;
}

.stButton > button[kind="primary"] p,
[data-testid="stBaseButton-primary"] p {
    color: #fff !important;
}

/* =====================================================
   HERO
   ===================================================== */

.hero {
    padding: 2.3rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1rem;

    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(230, 200, 102, .18),
            transparent 30%
        ),
        linear-gradient(
            120deg,
            var(--burgundy-dark),
            var(--burgundy-light)
        );

    box-shadow:
        0 12px 35px rgba(50, 4, 15, .14);
}

.hero h1 {
    color: #fff !important;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.03;
    margin: 0;
    letter-spacing: -0.025em;
}

.hero p {
    max-width: 790px;
    margin: .85rem 0 0;
    color: rgba(255, 255, 255, .82) !important;
    font-size: .98rem;
    line-height: 1.65;
}

.hero-badge {
    display: inline-block;
    margin-bottom: .8rem;
    padding: .34rem .7rem;
    border: 1px solid rgba(230, 200, 102, .4);
    border-radius: 999px;
    color: var(--gold);
    background: rgba(255, 255, 255, .06);
    font-size: .72rem;
    font-weight: 700;
}

/* =====================================================
   SECTIONS
   ===================================================== */

.section-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1.5rem 0 .15rem;
    color: var(--text);
}

.section-copy {
    color: var(--muted);
    font-size: .86rem;
    line-height: 1.65;
    max-width: 840px;
    margin-bottom: .9rem;
}

/* =====================================================
   KPI CARDS
   ===================================================== */

.kpi {
    height: 100%;
    min-height: 118px;
    padding: 1rem 1.1rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #fff;
    color: var(--text);
    box-shadow:
        0 3px 14px rgba(39, 28, 32, .035);
}

.kpi-label {
    color: var(--muted);
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .03em;
    text-transform: uppercase;
}

.kpi-value {
    margin-top: .25rem;
    color: var(--burgundy);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.1;
}

.kpi-sub {
    margin-top: .4rem;
    color: var(--muted);
    font-size: .74rem;
    line-height: 1.4;
}

/* =====================================================
   RESULT CARD
   ===================================================== */

.result {
    padding: 1.8rem 1.9rem;
    border-radius: 16px;
    color: #fff;

    background:
        radial-gradient(
            circle at 95% 0%,
            rgba(230, 200, 102, .16),
            transparent 28%
        ),
        linear-gradient(
            120deg,
            var(--burgundy-dark),
            var(--burgundy-light)
        );

    box-shadow:
        0 12px 28px rgba(50, 4, 15, .13);
}

.result .small {
    color: var(--gold);
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
}

.result .big {
    color: #fff;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: clamp(2.8rem, 5vw, 4.3rem);
    font-weight: 700;
    line-height: 1;
    margin-top: .25rem;
}

.result .caption {
    color: rgba(255, 255, 255, .8);
    font-size: .85rem;
    margin-top: .55rem;
    max-width: 650px;
    line-height: 1.55;
}

.pill {
    display: inline-block;
    margin-top: .95rem;
    padding: .38rem .82rem;
    border-radius: 999px;
    background: #fff;
    font-weight: 700;
    font-size: .78rem;
}

.track {
    position: relative;
    height: 9px;
    margin-top: 1.55rem;
    border-radius: 99px;
    background:
        linear-gradient(
            90deg,
            var(--green) 0%,
            #D7B64D 50%,
            var(--red) 100%
        );
}

.marker {
    position: absolute;
    top: 50%;
    width: 19px;
    height: 19px;
    border-radius: 50%;
    background: #fff;
    border: 3px solid var(--burgundy-light);
    transform: translate(-50%, -50%);
    box-shadow: 0 2px 8px rgba(0, 0, 0, .35);
}

.ticks {
    display: flex;
    justify-content: space-between;
    margin-top: .48rem;
    font-size: .7rem;
    color: rgba(255, 255, 255, .7);
}

/* =====================================================
   SECOND MODEL CARD
   ===================================================== */

.other {
    padding: 1.3rem 1.35rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #fff;
    height: 100%;
    color: var(--text);
}

.other .k {
    color: var(--muted);
    font-size: .76rem;
    font-weight: 600;
}

.other .v {
    margin-top: .25rem;
    color: var(--burgundy);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
}

.other .b {
    margin-top: .35rem;
    font-size: .79rem;
    font-weight: 700;
}

.other .line {
    height: 6px;
    margin-top: 1.1rem;
    border-radius: 99px;
    background: var(--border);
    overflow: hidden;
}

.other .line-fill {
    height: 100%;
    border-radius: 99px;
}

.other .foot {
    margin-top: .8rem;
    color: var(--muted);
    font-size: .72rem;
    line-height: 1.5;
}

/* =====================================================
   PROFILE CARDS
   ===================================================== */

.profile-card {
    padding: 1rem 1.1rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #fff;
    color: var(--text);
}

.profile-label {
    color: var(--muted);
    font-size: .7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .03em;
}

.profile-value {
    color: var(--text);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: .18rem;
}

/* =====================================================
   FACTORS
   ===================================================== */

.factor {
    display: flex;
    gap: .85rem;
    align-items: flex-start;
    padding: .85rem 1rem;
    margin-bottom: .55rem;
    border: 1px solid var(--border);
    border-radius: 11px;
    background: #fff;
}

.factor-icon {
    width: 30px;
    height: 30px;
    min-width: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-weight: 700;
    font-size: .85rem;
}

.factor-positive {
    background: var(--red-soft);
    color: var(--red);
}

.factor-negative {
    background: var(--green-soft);
    color: var(--green);
}

.factor-neutral {
    background: var(--blue-soft);
    color: var(--blue);
}

.factor-title {
    font-size: .81rem;
    font-weight: 700;
    color: var(--text);
}

.factor-num {
    color: var(--muted);
    font-weight: 500;
    margin-left: .35rem;
}

.factor-text {
    margin-top: .15rem;
    color: var(--muted);
    font-size: .74rem;
    line-height: 1.5;
}

/* =====================================================
   QUALITY
   ===================================================== */

.quality {
    padding: .8rem 1rem;
    border-radius: 11px;
    border: 1px solid var(--border);
    background: #fff;
    color: var(--text);
    font-size: .85rem;
}

.quality-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: .4rem;
    border-radius: 50%;
}

.quality-sub {
    color: var(--muted);
    margin-left: .4rem;
    font-size: .76rem;
}

/* =====================================================
   NOTE
   ===================================================== */

.note {
    padding: .9rem 1rem;
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    background: #FFFDF7;
    color: #4F464A;
    font-size: .82rem;
    line-height: 1.65;
}

/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 700px) {
    .hero {
        padding: 1.5rem 1.2rem;
    }

    .main .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }
}
"""


st.markdown(
    f"<style>:root {{{CSS_VARS}}}{CSS_RULES}</style>",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def html(block: str) -> None:
    """
    Render HTML safely.
    Lines are flattened so Markdown cannot interpret indented HTML as code.
    """
    flat = " ".join(
        line.strip()
        for line in textwrap.dedent(block).splitlines()
        if line.strip()
    )

    st.markdown(flat, unsafe_allow_html=True)


def section(title: str, copy: str = "") -> None:
    html(f'<div class="section-title">{title}</div>')

    if copy:
        html(f'<div class="section-copy">{copy}</div>')


def money(x: float) -> str:
    return f"&#36;{x:,.0f}"


def kpi(
    label_: str,
    value: str,
    sub: str = "",
    color: str | None = None,
) -> str:

    style = f' style="color:{color};"' if color else ""

    return (
        f'<div class="kpi">'
        f'<div class="kpi-label">{label_}</div>'
        f'<div class="kpi-value"{style}>{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def kpi_row(items: list[tuple]) -> None:
    columns = st.columns(len(items))

    for col, item in zip(columns, items):
        with col:
            html(kpi(*item))


def pcard(title: str, value: str) -> str:
    return (
        f'<div class="profile-card">'
        f'<div class="profile-label">{title}</div>'
        f'<div class="profile-value">{value}</div>'
        f'</div>'
    )


def pcard_row(items: list[tuple]) -> None:
    columns = st.columns(len(items))

    for col, (title, value) in zip(columns, items):
        with col:
            html(pcard(title, value))


# =========================================================
# MODEL PACKAGE
# =========================================================

REQUIRED_KEYS = {
    "logistic_model",
    "random_forest_model",
    "scaler",
    "feature_names",
    "comparison",
    "config",
}


@st.cache_resource(show_spinner="Loading trained models...")
def load_package(path: str = "credit_risk_models.pkl") -> dict:
    package = joblib.load(path)

    if not isinstance(package, dict):
        raise TypeError(
            f"{path} must contain a dictionary."
        )

    missing = REQUIRED_KEYS.difference(package.keys())

    if missing:
        raise KeyError(
            f"Model package is missing keys: {sorted(missing)}"
        )

    if not package["feature_names"]:
        raise ValueError(
            "The model package contains no feature_names."
        )

    return package


def normalize_comparison(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        result = data.copy()

    elif isinstance(data, (dict, list)):
        result = pd.DataFrame(data)

    else:
        raise TypeError(
            "Comparison must be a DataFrame, dict, or list."
        )

    if "Model" not in result.columns:
        raise ValueError(
            "Comparison data must contain a 'Model' column."
        )

    return result


try:
    package = load_package()
    comparison = normalize_comparison(
        package["comparison"]
    )

except FileNotFoundError:
    st.error(
        "Model file not found. Place "
        "`credit_risk_models.pkl` next to this app and reload."
    )
    st.stop()

except Exception as exc:
    st.error(
        "The trained model package could not be loaded."
    )

    st.code(
        f"{type(exc).__name__}: {exc}"
    )

    st.stop()


logistic_model = package["logistic_model"]
random_forest_model = package["random_forest_model"]
scaler = package["scaler"]
feature_names = list(package["feature_names"])
config = package["config"]

LR = "Logistic Regression"
RF = "Random Forest"

MODEL_NAMES = [LR, RF]


# =========================================================
# LABELS
# =========================================================

FEATURE_LABELS = {
    "age": "Age",
    "MonthlyIncome": "Monthly income",
    "NumberOfDependents": "Dependents",
    "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
    "DebtRatio": "Debt ratio",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days past due",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days past due",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "MonthlyIncome_missing": "Income missing flag",
    "total_past_due": "Total past-due events",
    "severe_delinquency_ratio": "Severe delinquency ratio",
    "total_credit_lines": "Total credit lines",
    "real_estate_ratio": "Real estate share",
    "log_income": "Income (log)",
    "income_per_credit_line": "Income per credit line",
    "high_debt_flag": "Debt ratio above 1",
    "log_debt_ratio": "Debt ratio (log)",
    "utilization_flag": "Utilization above 100%",
    "age_risk_score": "Age × past-due events",
}


def label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    if name.startswith("age_bin_"):
        return (
            "Age group: "
            + name.replace("age_bin_", "")
        )

    return name


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(inp: dict) -> pd.DataFrame:

    data = pd.DataFrame(
        {
            "age": [inp["age"]],
            "MonthlyIncome": [
                inp["monthly_income"]
            ],
            "NumberOfDependents": [
                inp["dependents"]
            ],
            "RevolvingUtilizationOfUnsecuredLines": [
                inp["utilization"]
            ],
            "DebtRatio": [
                inp["debt_ratio"]
            ],
            "NumberOfTime30-59DaysPastDueNotWorse": [
                inp["late_30"]
            ],
            "NumberOfTime60-89DaysPastDueNotWorse": [
                inp["late_60"]
            ],
            "NumberOfTimes90DaysLate": [
                inp["late_90"]
            ],
            "NumberOfOpenCreditLinesAndLoans": [
                inp["open_lines"]
            ],
            "NumberRealEstateLoansOrLines": [
                inp["real_estate"]
            ],
            "MonthlyIncome_missing": [0],
        }
    )

    d30 = data[
        "NumberOfTime30-59DaysPastDueNotWorse"
    ]

    d60 = data[
        "NumberOfTime60-89DaysPastDueNotWorse"
    ]

    d90 = data[
        "NumberOfTimes90DaysLate"
    ]

    open_lines = data[
        "NumberOfOpenCreditLinesAndLoans"
    ]

    re_lines = data[
        "NumberRealEstateLoansOrLines"
    ]

    data["total_past_due"] = (
        d30 + d60 + d90
    )

    data["severe_delinquency_ratio"] = (
        d90 /
        (data["total_past_due"] + 1)
    )

    data["total_credit_lines"] = (
        open_lines + re_lines
    )

    data["real_estate_ratio"] = (
        re_lines /
        (data["total_credit_lines"] + 1)
    )

    data["log_income"] = np.log1p(
        data["MonthlyIncome"].clip(lower=0)
    )

    data["income_per_credit_line"] = (
        data["MonthlyIncome"] /
        (open_lines + 1)
    )

    data["high_debt_flag"] = (
        data["DebtRatio"] > 1
    ).astype(int)

    data["log_debt_ratio"] = np.log1p(
        data["DebtRatio"].clip(lower=0)
    )

    data["utilization_flag"] = (
        data[
            "RevolvingUtilizationOfUnsecuredLines"
        ] > 1
    ).astype(int)

    data["age_bin"] = pd.cut(
        data["age"],
        bins=[0, 25, 35, 50, 65, 120],
        labels=[
            "young",
            "adult",
            "mid",
            "senior",
            "old",
        ],
        include_lowest=True,
    )

    data["age_risk_score"] = (
        data["age"] *
        data["total_past_due"]
    )

    data = pd.get_dummies(
        data,
        columns=["age_bin"],
        drop_first=True,
        dtype=int,
    )

    data = data.reindex(
        columns=feature_names,
        fill_value=0,
    )

    return (
        data
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def validate_model_inputs(
    X: pd.DataFrame,
) -> None:

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Model expects {len(feature_names)} "
            f"features, got {X.shape[1]}."
        )

    if list(X.columns) != feature_names:
        raise ValueError(
            "Feature order does not match "
            "the training order."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Generated features contain "
            "NaN or infinite values."
        )


# =========================================================
# PREDICTION
# =========================================================

def get_prediction(
    model_name: str,
    X: pd.DataFrame,
) -> tuple[float, int]:

    validate_model_inputs(X)

    if model_name == LR:
        X_model = scaler.transform(X)
        model = logistic_model

    elif model_name == RF:
        X_model = X
        model = random_forest_model

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    probability = float(
        model.predict_proba(X_model)[0, 1]
    )

    prediction = int(
        model.predict(X_model)[0]
    )

    return probability, prediction


# =========================================================
# MODEL EXPLANATION
# =========================================================

def logistic_contributions(
    X: pd.DataFrame,
) -> pd.Series | None:

    try:
        scaled = scaler.transform(X)[0]

        coefs = np.asarray(
            logistic_model.coef_
        )[0]

        return pd.Series(
            coefs * scaled,
            index=feature_names,
        )

    except Exception:
        return None


def rf_tree_stats(
    X: pd.DataFrame,
) -> dict | None:

    try:
        arr = X.to_numpy(dtype=float)

        probabilities = []

        for tree in random_forest_model.estimators_:

            row = tree.predict_proba(arr)[0]

            if len(row) > 1:
                probabilities.append(
                    float(row[1])
                )
            else:
                probabilities.append(0.0)

        probabilities = np.asarray(
            probabilities
        )

        counts, _ = np.histogram(
            probabilities,
            bins=10,
            range=(0, 1),
        )

        hist = pd.DataFrame(
            {"Trees": counts},
            index=[
                f"{i * 10}–{i * 10 + 10}%"
                for i in range(10)
            ],
        )

        return {
            "n": len(probabilities),
            "mean": probabilities.mean(),
            "std": probabilities.std(),
            "p05": np.percentile(
                probabilities,
                5,
            ),
            "p95": np.percentile(
                probabilities,
                95,
            ),
            "votes": int(
                (probabilities >= 0.5).sum()
            ),
            "hist": hist,
        }

    except Exception:
        return None


# =========================================================
# SENSITIVITY
# =========================================================

SENS_VARS = {
    "Revolving utilization": (
        "utilization",
        np.linspace(0, 2, 21),
        float,
    ),

    "Debt ratio": (
        "debt_ratio",
        np.linspace(0, 3, 21),
        float,
    ),

    "Monthly income": (
        "monthly_income",
        np.linspace(500, 15000, 21),
        float,
    ),

    "Age": (
        "age",
        np.arange(18, 91, 4),
        int,
    ),

    "30–59 days past due": (
        "late_30",
        np.arange(0, 11),
        int,
    ),

    "90+ days late": (
        "late_90",
        np.arange(0, 11),
        int,
    ),
}


def sensitivity_table(
    inp: dict,
    name: str,
) -> pd.DataFrame:

    key, grid, cast = SENS_VARS[name]

    rows = []

    for value in grid:

        trial = dict(inp)

        trial[key] = cast(value)

        X = create_features(trial)

        rows.append(
            {
                name: cast(value),
                LR: get_prediction(LR, X)[0],
                RF: get_prediction(RF, X)[0],
            }
        )

    return pd.DataFrame(rows).set_index(name)


# =========================================================
# RISK
# =========================================================

def classify_risk(
    p: float,
) -> tuple[str, str, str]:

    if p < 0.25:
        return (
            "Low risk",
            COLORS["green"],
            "Lower predicted probability of serious delinquency.",
        )

    if p < 0.50:
        return (
            "Moderate risk",
            COLORS["orange"],
            "Intermediate predicted probability of serious delinquency.",
        )

    return (
        "High risk",
        COLORS["red"],
        "Higher predicted probability of serious delinquency.",
    )


def risk_band_short(p: float) -> str:
    return classify_risk(p)[0]


def one_in(p: float) -> str:

    if p <= 0:
        return "—"

    if p < 1:
        return f"1 in {1 / p:,.0f}"

    return "1 in 1"


# =========================================================
# INPUT QUALITY
# =========================================================

def input_warnings(inp: dict) -> list[str]:

    notes = []

    if inp["monthly_income"] == 0:
        notes.append(
            "Monthly income is 0, so income-based "
            "features will be at their minimum."
        )

    if inp["debt_ratio"] > 1:
        notes.append(
            "Debt ratio is above 1: debt exceeds income."
        )

    if inp["utilization"] > 1:
        notes.append(
            "Revolving utilization is above 100%."
        )

    if (
        inp["age"] < 21
        and inp["late_90"] > 0
    ):
        notes.append(
            "Several past-due events at a very "
            "young age is an unusual combination."
        )

    return notes


def profile_quality(
    inp: dict,
) -> tuple[str, str]:

    issues = sum(
        [
            inp["monthly_income"] == 0,
            inp["debt_ratio"] > 5,
            inp["utilization"] > 5,
            inp["age"] < 18 or inp["age"] > 100,
        ]
    )

    if issues == 0:
        return (
            "Inputs valid",
            COLORS["green"],
        )

    if issues <= 2:
        return (
            "Review inputs",
            COLORS["orange"],
        )

    return (
        "Unusual profile",
        COLORS["red"],
    )


# =========================================================
# EXPLANATION
# =========================================================

PAST_DUE = {
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "total_past_due",
}


def explain_feature(
    f: str,
    v: float,
    c: float,
    inp: dict,
) -> str:

    effect = (
        "raises"
        if c > 0
        else "lowers"
    )

    if f == "RevolvingUtilizationOfUnsecuredLines":
        return (
            f"Utilization of "
            f"{inp['utilization']:.0%} "
            f"{effect} the estimated log-odds."
        )

    if f == "DebtRatio":
        return (
            f"A debt ratio of "
            f"{inp['debt_ratio']:.2f} "
            f"{effect} the estimated log-odds."
        )

    if f in PAST_DUE:
        if v > 0:
            return (
                f"{v:.0f} recorded event(s); "
                f"this {effect} the estimated log-odds."
            )

        return (
            "No events recorded for this measure."
        )

    if f in {
        "high_debt_flag",
        "utilization_flag",
    }:

        if v > 0:
            return (
                "Flag is on: the threshold "
                "is exceeded."
            )

        return (
            "Flag is off: the threshold "
            "is not exceeded."
        )

    return (
        f"This feature {effect} the estimated "
        f"log-odds for this applicant."
    )


def build_factor_rows(
    contrib: pd.Series,
    X: pd.DataFrame,
    inp: dict,
    n: int = 8,
) -> list[dict]:

    ranked = (
        contrib.abs()
        .sort_values(ascending=False)
        .head(n)
        .index
    )

    rows = []

    for f in ranked:

        c = float(contrib[f])
        v = float(X.iloc[0][f])

        rows.append(
            {
                "label": label(f),
                "contribution": c,
                "direction": (
                    "positive"
                    if c > 0
                    else "negative"
                    if c < 0
                    else "neutral"
                ),
                "explanation": explain_feature(
                    f,
                    v,
                    c,
                    inp,
                ),
            }
        )

    return rows


def model_params(
    model,
    keys: list[str],
) -> pd.DataFrame:

    try:
        params = model.get_params()

        rows = [
            (k, str(params[k]))
            for k in keys
            if k in params
        ]

    except Exception:
        rows = []

    return pd.DataFrame(
        rows,
        columns=["Parameter", "Value"],
    )


# =========================================================
# STATE / PRESETS
# =========================================================

DEFAULTS = {
    "applicant_age": 41,
    "applicant_monthly_income": 5000.0,
    "applicant_dependents": 1,
    "applicant_revolving_utilization": 0.35,
    "applicant_debt_ratio": 0.40,
    "applicant_open_credit_lines": 8,
    "applicant_real_estate_loans": 1,
    "delinq_30_59": 0,
    "delinq_60_89": 0,
    "delinq_90_plus": 0,
}


PRESETS = {
    "Typical applicant": DEFAULTS,

    "Established, low utilization": {
        "applicant_age": 52,
        "applicant_monthly_income": 9000.0,
        "applicant_dependents": 0,
        "applicant_revolving_utilization": 0.08,
        "applicant_debt_ratio": 0.20,
        "applicant_open_credit_lines": 9,
        "applicant_real_estate_loans": 2,
        "delinq_30_59": 0,
        "delinq_60_89": 0,
        "delinq_90_plus": 0,
    },

    "Stretched, repeated late payments": {
        "applicant_age": 29,
        "applicant_monthly_income": 2200.0,
        "applicant_dependents": 3,
        "applicant_revolving_utilization": 1.40,
        "applicant_debt_ratio": 1.60,
        "applicant_open_credit_lines": 4,
        "applicant_real_estate_loans": 0,
        "delinq_30_59": 3,
        "delinq_60_89": 2,
        "delinq_90_plus": 2,
    },
}


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

for key, value in DEFAULTS.items():
    st.session_state.setdefault(
        key,
        value,
    )

st.session_state.setdefault(
    "assessment_history",
    [],
)

st.session_state.setdefault(
    "result",
    None,
)


# =========================================================
# CALLBACKS
# =========================================================

def apply_preset(name: str) -> None:
    """
    Callback used by preset buttons.

    IMPORTANT:
    Session-state values are changed here before the
    widgets are instantiated during the subsequent rerun.
    """

    preset = PRESETS[name]

    for key, value in preset.items():
        st.session_state[key] = value

    st.session_state["result"] = None


def reset_applicant() -> None:
    """
    Reset applicant values.

    This function is intentionally used ONLY as a button
    callback. It must not be called later in the script
    after widgets have been instantiated.
    """

    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.session_state["result"] = None


def clear_history() -> None:
    st.session_state["assessment_history"] = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("### Model controls")

    model_name = st.selectbox(
        "Headline model",
        MODEL_NAMES,
        key="sidebar_prediction_model",
        help=(
            "Both models are always evaluated. "
            "This selects which model is displayed "
            "as the headline estimate."
        ),
    )

    st.divider()

    st.markdown("### Risk bands")

    st.markdown(
        "🟢 **Low**: below 25%  \n"
        "🟠 **Moderate**: 25% to 50%  \n"
        "🔴 **High**: 50% and above"
    )

    st.divider()

    st.markdown("### Target")

    st.markdown("`SeriousDlqin2yrs`")

    st.caption(
        "Serious delinquency within two years."
    )

    st.divider()

    st.markdown("### Session")

    st.metric(
        "Assessments this session",
        len(
            st.session_state[
                "assessment_history"
            ]
        ),
    )

    if st.session_state["assessment_history"]:

        st.button(
            "Clear assessment history",
            key="clear_history",
            on_click=clear_history,
        )

    st.divider()

    st.caption(
        "Academic demonstration only. "
        "Not for lending, credit approval, "
        "underwriting, or any real financial decision."
    )


# =========================================================
# HERO
# =========================================================

html(
    """
    <div class="hero">
        <div class="hero-badge">
            Machine learning · Credit analytics
        </div>

        <h1>
            Credit Risk Analytics
        </h1>

        <p>
            Explore how two trained models estimate the probability
            of serious delinquency within two years. Compare model
            outputs, inspect engineered features, test what-if
            scenarios, and review how the models were evaluated.
        </p>
    </div>
    """
)


# =========================================================
# TABS
# =========================================================

(
    tab_assess,
    tab_perf,
    tab_history,
    tab_about,
) = st.tabs(
    [
        "Assess an applicant",
        "Model performance",
        "Assessment history",
        "How it works",
    ]
)


# =========================================================
# TAB 1 — ASSESS
# =========================================================

with tab_assess:

    section(
        "Applicant profile",
        "Start with a sample profile or enter your own values. "
        "Derived features are calculated automatically to match "
        "the structure the stored models expect.",
    )

    # -----------------------------------------------------
    # PRESETS
    # -----------------------------------------------------

    preset_cols = st.columns(
        len(PRESETS) + 1
    )

    for col, name in zip(
        preset_cols[:-1],
        PRESETS,
    ):

        col.button(
            name,
            key=f"preset_{name}",
            on_click=apply_preset,
            args=(name,),
        )

    preset_cols[-1].button(
        "Reset",
        key="reset_applicant",
        on_click=reset_applicant,
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(
        3,
        gap="large",
    )

    with c1:

        with st.container(border=True):

            st.markdown(
                "**Personal profile**"
            )

            st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                step=1,
                key="applicant_age",
            )

            st.number_input(
                "Monthly income",
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="applicant_monthly_income",
            )

            st.number_input(
                "Number of dependents",
                min_value=0,
                max_value=20,
                step=1,
                key="applicant_dependents",
            )

    with c2:

        with st.container(border=True):

            st.markdown(
                "**Credit profile**"
            )

            st.number_input(
                "Revolving utilization",
                min_value=0.0,
                max_value=20.0,
                step=0.01,
                format="%.2f",
                key="applicant_revolving_utilization",
                help=(
                    "Balance on revolving credit divided "
                    "by available limits. 1.00 is 100%."
                ),
            )

            st.number_input(
                "Debt ratio",
                min_value=0.0,
                max_value=20.0,
                step=0.01,
                format="%.2f",
                key="applicant_debt_ratio",
                help=(
                    "Monthly debt payments divided "
                    "by monthly income."
                ),
            )

            st.number_input(
                "Open credit lines and loans",
                min_value=0,
                max_value=100,
                step=1,
                key="applicant_open_credit_lines",
            )

    with c3:

        with st.container(border=True):

            st.markdown(
                "**Payment history**"
            )

            st.number_input(
                "30–59 days past due",
                min_value=0,
                max_value=50,
                step=1,
                key="delinq_30_59",
            )

            st.number_input(
                "60–89 days past due",
                min_value=0,
                max_value=50,
                step=1,
                key="delinq_60_89",
            )

            st.number_input(
                "90+ days late",
                min_value=0,
                max_value=50,
                step=1,
                key="delinq_90_plus",
            )

    with st.container(border=True):

        st.markdown(
            "**Property profile**"
        )

        st.number_input(
            "Real estate loans and lines",
            min_value=0,
            max_value=50,
            step=1,
            key="applicant_real_estate_loans",
        )

    # -----------------------------------------------------
    # CURRENT INPUTS
    # -----------------------------------------------------

    current_inputs = {
        "age": st.session_state[
            "applicant_age"
        ],

        "monthly_income": st.session_state[
            "applicant_monthly_income"
        ],

        "dependents": st.session_state[
            "applicant_dependents"
        ],

        "utilization": st.session_state[
            "applicant_revolving_utilization"
        ],

        "debt_ratio": st.session_state[
            "applicant_debt_ratio"
        ],

        "open_lines": st.session_state[
            "applicant_open_credit_lines"
        ],

        "real_estate": st.session_state[
            "applicant_real_estate_loans"
        ],

        "late_30": st.session_state[
            "delinq_30_59"
        ],

        "late_60": st.session_state[
            "delinq_60_89"
        ],

        "late_90": st.session_state[
            "delinq_90_plus"
        ],
    }

    # -----------------------------------------------------
    # INPUT QUALITY
    # -----------------------------------------------------

    quality_text, quality_color = (
        profile_quality(current_inputs)
    )

    html(
        f"""
        <div class="quality">
            <span
                class="quality-dot"
                style="background:{quality_color};"
            ></span>

            <strong>{quality_text}</strong>

            <span class="quality-sub">
                All fields are within the permitted ranges.
            </span>
        </div>
        """
    )

    for note in input_warnings(
        current_inputs
    ):
        st.warning(
            note,
            icon="⚠️",
        )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    _, mid, _ = st.columns(
        [1, 1.3, 1]
    )

    with mid:

        predict_clicked = st.button(
            "Estimate credit risk",
            key="predict_credit_risk",
            type="primary",
        )

    if predict_clicked:

        try:

            X_input = create_features(
                current_inputs
            )

            preds = {
                name: get_prediction(
                    name,
                    X_input,
                )
                for name in MODEL_NAMES
            }

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            st.session_state["result"] = {
                "inputs": dict(
                    current_inputs
                ),
                "features": X_input.copy(),
                "preds": preds,
                "timestamp": timestamp,
            }

            headline_probability = preds[
                model_name
            ][0]

            st.session_state[
                "assessment_history"
            ].insert(
                0,
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "headline_probability":
                        headline_probability,
                    "logistic_probability":
                        preds[LR][0],
                    "random_forest_probability":
                        preds[RF][0],
                    "risk_band":
                        risk_band_short(
                            headline_probability
                        ),
                    "age":
                        current_inputs["age"],
                    "income":
                        current_inputs[
                            "monthly_income"
                        ],
                    "debt_ratio":
                        current_inputs[
                            "debt_ratio"
                        ],
                    "utilization":
                        current_inputs[
                            "utilization"
                        ],
                },
            )

            st.session_state[
                "assessment_history"
            ] = st.session_state[
                "assessment_history"
            ][:25]

        except Exception as exc:

            st.error(
                "The prediction could not be generated."
            )

            st.code(
                f"{type(exc).__name__}: {exc}"
            )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    result = st.session_state.get(
        "result"
    )

    if not result:

        html(
            """
            <div class="note">
                <strong>No estimate yet.</strong>
                Choose a sample profile or enter applicant
                values, then select
                <strong>Estimate credit risk</strong>.
            </div>
            """
        )

    else:

        ri = result["inputs"]
        RX = result["features"]

        section(
            "Assessment result",
            "The headline result follows the model chosen "
            "in the sidebar. Both models are always evaluated "
            "so you can compare them.",
        )

        if ri != current_inputs:

            st.info(
                "Inputs have changed since this estimate. "
                "Select “Estimate credit risk” to update it."
            )

        probability, prediction = (
            result["preds"][model_name]
        )

        (
            risk_label,
            risk_color,
            risk_description,
        ) = classify_risk(
            probability
        )

        other_name = next(
            name
            for name in MODEL_NAMES
            if name != model_name
        )

        other_prob, _ = (
            result["preds"][other_name]
        )

        (
            other_label,
            other_color,
            _,
        ) = classify_risk(
            other_prob
        )

        gap = abs(
            probability - other_prob
        )

        same_band = (
            risk_label == other_label
        )

        p_lr = result["preds"][LR][0]
        p_rf = result["preds"][RF][0]

        avg_p = (
            p_lr + p_rf
        ) / 2

        (
            avg_label,
            avg_color,
            _,
        ) = classify_risk(avg_p)

        # -------------------------------------------------
        # KPI ROW
        # -------------------------------------------------

        kpi_row(
            [
                (
                    "Headline probability",
                    f"{probability:.1%}",
                    model_name,
                ),

                (
                    "Risk band",
                    risk_label,
                    "Based on the app's thresholds",
                    risk_color,
                ),

                (
                    "Model gap",
                    f"{gap:.1%}",
                    "Absolute probability difference",
                ),

                (
                    "Model agreement",
                    (
                        "Same band"
                        if same_band
                        else "Different bands"
                    ),
                    f"{model_name} vs {other_name}",
                    (
                        COLORS["green"]
                        if same_band
                        else COLORS["orange"]
                    ),
                ),
            ]
        )

        st.write("")

        # -------------------------------------------------
        # ADDITIONAL METRICS
        # -------------------------------------------------

        kpi_row(
            [
                (
                    "Ensemble average",
                    f"{avg_p:.1%}",
                    f"Mean of both models · {avg_label}",
                    avg_color,
                ),

                (
                    "Per 100 similar",
                    f"{probability * 100:.1f}",
                    "Reading if perfectly calibrated",
                ),

                (
                    "Odds",
                    one_in(probability),
                    "Approximate one-in-N frequency",
                ),

                (
                    "Distance from 50%",
                    f"{(probability - 0.5) * 100:+.1f} pts",
                    "Negative means below the high-risk line",
                ),
            ]
        )

        st.write("")

        # -------------------------------------------------
        # HEADLINE RESULT
        # -------------------------------------------------

        left, right = st.columns(
            [2, 1],
            gap="medium",
        )

        with left:

            marker = min(
                max(probability * 100, 2),
                98,
            )

            html(
                f"""
                <div class="result">

                    <div class="small">
                        Predicted probability · {model_name}
                    </div>

                    <div class="big">
                        {probability:.1%}
                    </div>

                    <div class="caption">
                        {risk_description}
                    </div>

                    <div
                        class="pill"
                        style="color:{risk_color};"
                    >
                        {risk_label}
                    </div>

                    <div class="track">
                        <div
                            class="marker"
                            style="left:{marker}%;"
                        ></div>
                    </div>

                    <div class="ticks">
                        <span>0%</span>
                        <span>25%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>

                </div>
                """
            )

        with right:

            width = min(
                max(other_prob * 100, 2),
                98,
            )

            html(
                f"""
                <div class="other">

                    <div class="k">
                        Second opinion · {other_name}
                    </div>

                    <div class="v">
                        {other_prob:.1%}
                    </div>

                    <div
                        class="b"
                        style="color:{other_color};"
                    >
                        {other_label}
                    </div>

                    <div class="line">
                        <div
                            class="line-fill"
                            style="
                                width:{width}%;
                                background:{other_color};
                            "
                        ></div>
                    </div>

                    <div class="foot">
                        Independent output for the same
                        engineered feature row.
                    </div>

                </div>
                """
            )

        if same_band:

            st.success(
                f"Both models place this applicant "
                f"in the same band: "
                f"{risk_label.lower()}.",
                icon="✅",
            )

        else:

            st.warning(
                f"The models disagree: "
                f"{model_name} says "
                f"{risk_label.lower()}, "
                f"{other_name} says "
                f"{other_label.lower()}. "
                f"The gap is {gap:.1%}.",
                icon="⚖️",
            )

        # -------------------------------------------------
        # SNAPSHOT
        # -------------------------------------------------

        section(
            "Applicant snapshot",
            "The profile used for this estimate, "
            "plus values derived from it.",
        )

        total_late = (
            ri["late_30"]
            + ri["late_60"]
            + ri["late_90"]
        )

        pcard_row(
            [
                (
                    "Age",
                    f"{ri['age']:.0f}",
                ),

                (
                    "Monthly income",
                    money(
                        ri["monthly_income"]
                    ),
                ),

                (
                    "Utilization",
                    f"{ri['utilization']:.2f}",
                ),

                (
                    "Debt ratio",
                    f"{ri['debt_ratio']:.2f}",
                ),

                (
                    "Past-due events",
                    f"{total_late:.0f}",
                ),
            ]
        )

        st.write("")

        pcard_row(
            [
                (
                    "Est. monthly debt",
                    money(
                        ri["debt_ratio"]
                        * ri["monthly_income"]
                    ),
                ),

                (
                    "Income per credit line",
                    money(
                        ri["monthly_income"]
                        / (
                            ri["open_lines"]
                            + 1
                        )
                    ),
                ),

                (
                    "Total credit lines",
                    f"{ri['open_lines'] + ri['real_estate']:.0f}",
                ),

                (
                    "Severe late share",
                    f"{ri['late_90'] / (total_late + 1):.2f}",
                ),

                (
                    "Dependents",
                    f"{ri['dependents']:.0f}",
                ),
            ]
        )

        # -------------------------------------------------
        # DIAGNOSTICS
        # -------------------------------------------------

        contrib = logistic_contributions(
            RX
        )

        tree = rf_tree_stats(
            RX
        )

        section(
            "Model diagnostics",
            "How each model arrived at its number. "
            "The Logistic Regression view is exact for "
            "this row; the Random Forest view shows how "
            "much its individual trees disagree.",
        )

        dl, dr = st.columns(
            2,
            gap="large",
        )

        with dl:

            with st.container(
                border=True
            ):

                st.markdown(
                    "**Logistic Regression**"
                )

                if contrib is not None:

                    try:
                        intercept = float(
                            np.asarray(
                                logistic_model.intercept_
                            )[0]
                        )

                    except Exception:
                        intercept = 0.0

                    total = float(
                        contrib.sum()
                    )

                    logit = (
                        intercept + total
                    )

                    up = int(
                        (contrib > 0).sum()
                    )

                    down = int(
                        (contrib < 0).sum()
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Intercept",
                        f"{intercept:+.3f}",
                    )

                    b.metric(
                        "Sum of contributions",
                        f"{total:+.3f}",
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Total log-odds",
                        f"{logit:+.3f}",
                    )

                    b.metric(
                        "Drivers up / down",
                        f"{up} / {down}",
                    )

                    calculated_probability = (
                        1 /
                        (
                            1 +
                            np.exp(-logit)
                        )
                    )

                    st.caption(
                        "Probability = "
                        "1 / (1 + e^−log-odds) = "
                        f"{calculated_probability:.2%}. "
                        f"Model output: {p_lr:.2%}."
                    )

                else:

                    st.info(
                        "Contributions are unavailable "
                        "for the stored Logistic Regression model."
                    )

        with dr:

            with st.container(
                border=True
            ):

                st.markdown(
                    "**Random Forest**"
                )

                if tree is not None:

                    a, b = st.columns(2)

                    a.metric(
                        "Mean tree probability",
                        f"{tree['mean']:.1%}",
                    )

                    b.metric(
                        "Tree disagreement (std)",
                        f"{tree['std']:.3f}",
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Trees voting delinquent",
                        f"{tree['votes']} / {tree['n']}",
                    )

                    b.metric(
                        "Middle 90% of trees",
                        f"{tree['p05']:.0%} – {tree['p95']:.0%}",
                    )

                    st.bar_chart(
                        tree["hist"],
                        height=170,
                        color=COLORS[
                            "burgundy_light"
                        ],
                    )

                    st.caption(
                        "Distribution of individual tree "
                        "probabilities for this applicant."
                    )

                else:

                    st.info(
                        "Per-tree statistics are unavailable "
                        "for the stored Random Forest model."
                    )

        # -------------------------------------------------
        # DRIVERS
        # -------------------------------------------------

        if contrib is not None:

            section(
                "What influenced the Logistic Regression estimate",
                "Each value is a feature's push on the log-odds "
                "for this applicant. Positive values raise the "
                "estimate, negative values lower it. These are "
                "model explanations, not causal effects.",
            )

            factors = build_factor_rows(
                contrib,
                RX,
                ri,
                n=8,
            )

            fl, fr = st.columns(
                [1.15, 1],
                gap="large",
            )

            with fl:

                for fac in factors:

                    icon, css = {
                        "positive": (
                            "↑",
                            "factor-positive",
                        ),

                        "negative": (
                            "↓",
                            "factor-negative",
                        ),

                        "neutral": (
                            "•",
                            "factor-neutral",
                        ),
                    }.get(
                        fac["direction"],
                        (
                            "•",
                            "factor-neutral",
                        ),
                    )

                    html(
                        f"""
                        <div class="factor">

                            <div
                                class="factor-icon {css}"
                            >
                                {icon}
                            </div>

                            <div>

                                <div class="factor-title">
                                    {fac["label"]}

                                    <span
                                        class="factor-num"
                                    >
                                        {fac["contribution"]:+.3f}
                                    </span>
                                </div>

                                <div class="factor-text">
                                    {fac["explanation"]}
                                </div>

                            </div>

                        </div>
                        """
                    )

            with fr:

                chart = pd.DataFrame(
                    {
                        "Effect on risk": [
                            f["contribution"]
                            for f in factors
                        ]
                    },
                    index=[
                        f["label"]
                        for f in factors
                    ],
                ).sort_values(
                    "Effect on risk"
                )

                st.bar_chart(
                    chart,
                    horizontal=True,
                    color=COLORS[
                        "burgundy_light"
                    ],
                )

                st.caption(
                    "Log-odds contribution of the "
                    "eight strongest drivers."
                )

        # -------------------------------------------------
        # WHAT-IF
        # -------------------------------------------------

        section(
            "What-if analysis",
            "Change one input while holding the rest "
            "of this applicant's profile fixed, and see "
            "how both models respond.",
        )

        sens_name = st.selectbox(
            "Input to vary",
            list(SENS_VARS),
            key="sens_var",
        )

        try:

            sens = sensitivity_table(
                ri,
                sens_name,
            )

            st.line_chart(
                sens,
                height=320,
                color=[
                    COLORS["burgundy_light"],
                    COLORS["gold"],
                ],
            )

            lo = sens.iloc[0]
            hi = sens.iloc[-1]

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                f"{LR} at lowest value",
                f"{lo[LR]:.1%}",
            )

            s2.metric(
                f"{LR} at highest value",
                f"{hi[LR]:.1%}",
            )

            s3.metric(
                f"{RF} at lowest value",
                f"{lo[RF]:.1%}",
            )

            s4.metric(
                f"{RF} at highest value",
                f"{hi[RF]:.1%}",
            )

            st.caption(
                f"Range shown: "
                f"{sens.index[0]:,.2f} to "
                f"{sens.index[-1]:,.2f}. "
                "Derived features are recalculated "
                "at every point."
            )

        except Exception as exc:

            st.warning(
                "What-if analysis could not be computed: "
                f"{exc}"
            )

        # -------------------------------------------------
        # MODEL COMPARISON
        # -------------------------------------------------

        section(
            "Model comparison for this applicant",
            "Both models receive the same engineered row. "
            "Logistic Regression uses the stored scaler; "
            "Random Forest uses the raw engineered values.",
        )

        rows = []

        for name in MODEL_NAMES:

            p, pred = result[
                "preds"
            ][name]

            rows.append(
                {
                    "Model": name,
                    "Probability": f"{p:.2%}",
                    "Risk band":
                        risk_band_short(p),
                    "Predicted class":
                        "1" if pred else "0",
                    "Odds": one_in(p),
                }
            )

        rows.append(
            {
                "Model": "Ensemble average",
                "Probability": f"{avg_p:.2%}",
                "Risk band": avg_label,
                "Predicted class":
                    "1"
                    if avg_p >= 0.5
                    else "0",
                "Odds": one_in(avg_p),
            }
        )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "View engineered feature values"
        ):

            disp = RX.T.copy()

            disp.columns = ["Value"]

            disp.index = [
                label(feature)
                for feature in disp.index
            ]

            disp.index.name = "Feature"

            st.dataframe(
                disp,
                height=380,
                use_container_width=True,
            )

        with st.expander(
            "Prediction notes and limitations"
        ):

            st.markdown(
                "- The probability is a **model output**, "
                "not a guaranteed outcome.\n"
                "- Risk bands are presentation thresholds "
                "defined by this application.\n"
                "- Logistic Regression contributions describe "
                "model behaviour for this row, not causal "
                "relationships.\n"
                "- What-if curves show model sensitivity, "
                "not what would happen to a real applicant.\n"
                "- Random Forest tree spread reflects model "
                "variance, not calibrated uncertainty.\n"
                "- This app must not be used for real lending, "
                "underwriting, or credit decisions."
            )

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------

        section(
            "Export assessment",
            "Download this assessment for reporting "
            "or further analysis.",
        )

        report = pd.DataFrame(
            [
                {
                    **ri,

                    **{
                        f"{name} probability": p
                        for name, (p, _)
                        in result["preds"].items()
                    },

                    **{
                        f"{name} class": c
                        for name, (_, c)
                        in result["preds"].items()
                    },

                    "ensemble_probability":
                        avg_p,

                    "headline_model":
                        model_name,

                    "headline_risk_band":
                        risk_label,

                    "headline_probability":
                        probability,

                    "timestamp":
                        result["timestamp"],
                }
            ]
        )

        report_json = json.dumps(
            {
                "timestamp":
                    result["timestamp"],

                "headline_model":
                    model_name,

                "headline_probability":
                    probability,

                "headline_risk_band":
                    risk_label,

                "ensemble_probability":
                    avg_p,

                "inputs":
                    ri,

                "predictions": {
                    name: {
                        "probability": p,
                        "class": c,
                        "risk_band":
                            risk_band_short(p),
                    }

                    for name, (p, c)
                    in result["preds"].items()
                },
            },
            indent=2,
            default=str,
        )

        e1, e2, e3 = st.columns(3)

        with e1:

            st.download_button(
                "Download CSV",
                report
                .to_csv(index=False)
                .encode("utf-8"),
                file_name=(
                    "credit_risk_assessment.csv"
                ),
                mime="text/csv",
            )

        with e2:

            st.download_button(
                "Download JSON",
                report_json.encode("utf-8"),
                file_name=(
                    "credit_risk_assessment.json"
                ),
                mime="application/json",
            )

        with e3:

            # =================================================
            # CRITICAL FIX
            #
            # DO NOT:
            #
            # if st.button(...):
            #     reset_applicant()
            #     st.rerun()
            #
            # The callback runs before the next script
            # execution, so applicant widgets can safely
            # receive the reset values.
            # =================================================

            st.button(
                "Start new assessment",
                key="new_assessment",
                on_click=reset_applicant,
            )


# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

with tab_perf:

    section(
        "Model performance",
        "Metrics stored with the project package, measured "
        "on the evaluation data used during model development. "
        "They describe recorded results, not future performance.",
    )

    METRIC_COLUMNS = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "roc_auc",
        "auc",
    }

    numeric = comparison.copy()

    metric_cols = [
        col
        for col in numeric.columns
        if col.lower() in METRIC_COLUMNS
    ]

    for col in metric_cols:
        numeric[col] = pd.to_numeric(
            numeric[col],
            errors="coerce",
        )

    if metric_cols:

        # -------------------------------------------------
        # SCORECARD
        # -------------------------------------------------

        st.markdown(
            "#### Model evaluation metrics"
        )

        cards = []

        for col in metric_cols:

            series = (
                numeric
                .set_index("Model")[col]
                .dropna()
            )

            if series.empty:
                continue

            best_model = series.idxmax()
            best_value = series.max()

            cards.append(
                (
                    col.replace(
                        "_",
                        " "
                    ).title(),

                    f"{best_value:.2%}",

                    f"{best_model} · "
                    f"recorded evaluation value",
                )
            )

        for i in range(
            0,
            len(cards),
            3,
        ):

            kpi_row(
                cards[i:i + 3]
            )

            st.write("")

        # -------------------------------------------------
        # MODEL TABLE
        # -------------------------------------------------

        st.markdown(
            "#### Scorecard by model"
        )

        for _, row in numeric.iterrows():

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{row['Model']}**"
                )

                vals = [
                    (metric, row[metric])
                    for metric in metric_cols
                    if pd.notna(row[metric])
                ]

                for j in range(
                    0,
                    len(vals),
                    3,
                ):

                    cols = st.columns(3)

                    for cc, (metric, value) in zip(
                        cols,
                        vals[j:j + 3],
                    ):

                        cc.metric(
                            metric.replace(
                                "_",
                                " "
                            ).title(),
                            f"{value:.2%}",
                        )

        # -------------------------------------------------
        # SPREAD TABLE
        # -------------------------------------------------

        spread_rows = []

        for col in metric_cols:

            series = (
                numeric
                .set_index("Model")[col]
                .dropna()
            )

            if len(series) > 1:

                spread_rows.append(
                    {
                        "Metric": col,

                        "Recorded highest":
                            f"{series.max():.2%}",

                        "Recorded lowest":
                            f"{series.min():.2%}",

                        "Difference (pts)":
                            f"{(
                                series.max()
                                - series.min()
                            ) * 100:.2f}",
                    }
                )

        if spread_rows:

            st.markdown(
                "#### Metric differences"
            )

            st.dataframe(
                pd.DataFrame(
                    spread_rows
                ),
                hide_index=True,
                use_container_width=True,
            )

    # -----------------------------------------------------
    # FULL TABLE
    # -----------------------------------------------------

    st.markdown(
        "#### Full evaluation table"
    )

    display = numeric.copy()

    for col in metric_cols:

        display[col] = numeric[col].map(
            lambda x:
                f"{x:.2%}"
                if pd.notna(x)
                else "N/A"
        )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
    )

    if metric_cols:

        chartable = (
            numeric
            .set_index("Model")[metric_cols]
            .dropna(how="all")
        )

        if not chartable.empty:

            st.markdown(
                "#### Metric comparison"
            )

            st.bar_chart(
                chartable.T,
                height=360,
            )

            st.caption(
                "Stored evaluation metrics per model. "
                "This does not reflect deployment cost, "
                "fairness, calibration, or population change."
            )

    # -----------------------------------------------------
    # MODEL DETAILS
    # -----------------------------------------------------

    section(
        "Model details",
        "Configuration of the two stored models.",
    )

    ml, mr = st.columns(
        2,
        gap="large",
    )

    with ml:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Logistic Regression**"
            )

            st.dataframe(
                model_params(
                    logistic_model,
                    [
                        "C",
                        "penalty",
                        "solver",
                        "max_iter",
                        "class_weight",
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )

            try:

                st.metric(
                    "Coefficients",
                    int(
                        np.asarray(
                            logistic_model.coef_
                        ).shape[1]
                    ),
                )

            except Exception:
                pass

    with mr:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Random Forest**"
            )

            st.dataframe(
                model_params(
                    random_forest_model,
                    [
                        "n_estimators",
                        "max_depth",
                        "min_samples_leaf",
                        "min_samples_split",
                        "max_features",
                        "class_weight",
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )

            try:

                depths = [
                    tree.get_depth()
                    for tree
                    in random_forest_model.estimators_
                ]

                a, b = st.columns(2)

                a.metric(
                    "Average tree depth",
                    f"{np.mean(depths):.1f}",
                )

                b.metric(
                    "Deepest tree",
                    int(np.max(depths)),
                )

            except Exception:
                pass

    # -----------------------------------------------------
    # LOGISTIC COEFFICIENTS
    # -----------------------------------------------------

    try:

        coefs = pd.Series(
            np.asarray(
                logistic_model.coef_
            )[0],
            index=[
                label(feature)
                for feature
                in feature_names
            ],
        )

        section(
            "Logistic Regression coefficients",
            "Coefficients on scaled features, so their sizes "
            "are comparable. Positive values raise the estimated "
            "log-odds of delinquency.",
        )

        top = (
            coefs
            .reindex(
                coefs.abs()
                .sort_values(
                    ascending=False
                )
                .head(15)
                .index
            )
            .sort_values()
        )

        st.bar_chart(
            top.to_frame(
                "Coefficient"
            ),
            horizontal=True,
            color=COLORS[
                "burgundy_light"
            ],
        )

    except Exception:
        pass

    # -----------------------------------------------------
    # RANDOM FOREST IMPORTANCE
    # -----------------------------------------------------

    section(
        "Random Forest feature importance",
        "How much each feature contributes to the forest's "
        "split-based impurity reduction. It is not a causal "
        "effect and does not explain an individual applicant.",
    )

    try:

        importances = (
            random_forest_model
            .feature_importances_
        )

        if len(importances) == len(
            feature_names
        ):

            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": [
                            label(feature)
                            for feature
                            in feature_names
                        ],

                        "Importance":
                            importances,
                    }
                )
                .sort_values(
                    "Importance",
                    ascending=False,
                )
            )

            st.bar_chart(
                importance_df
                .head(15)
                .set_index("Feature")
                .sort_values(
                    "Importance"
                ),
                horizontal=True,
                color=COLORS[
                    "burgundy_light"
                ],
            )

            with st.expander(
                "Top 10 features as a table"
            ):

                top10 = (
                    importance_df
                    .head(10)
                    .copy()
                )

                top10["Importance"] = (
                    top10["Importance"]
                    .map(
                        lambda x:
                            f"{x:.4f}"
                    )
                )

                st.dataframe(
                    top10,
                    hide_index=True,
                    use_container_width=True,
                )

        else:

            st.warning(
                "Feature importance length does not "
                "match feature_names."
            )

    except AttributeError:

        st.info(
            "The stored Random Forest model does not "
            "expose feature_importances_."
        )

    except Exception as exc:

        st.warning(
            "Could not display feature importance: "
            f"{exc}"
        )

    with st.expander(
        "Stored project configuration"
    ):

        st.write(config)


# =========================================================
# TAB 3 — HISTORY
# =========================================================

with tab_history:

    section(
        "Assessment history",
        "Estimates generated in this browser session. "
        "They live in Streamlit session state and are "
        "not saved by the app.",
    )

    history = st.session_state[
        "assessment_history"
    ]

    if not history:

        html(
            """
            <div class="note">
                No assessments yet. Generate an estimate
                on the first tab to see it here.
            </div>
            """
        )

    else:

        hdf = pd.DataFrame(
            history
        )

        h1, h2, h3, h4 = st.columns(4)

        h1.metric(
            "Assessments",
            len(hdf),
        )

        h2.metric(
            "Average probability",
            f"{hdf['headline_probability'].mean():.1%}",
        )

        h3.metric(
            "Highest probability",
            f"{hdf['headline_probability'].max():.1%}",
        )

        h4.metric(
            "High-risk share",
            f"{(
                hdf['risk_band'] == 'High risk'
            ).mean():.0%}",
        )

        shown = hdf.copy()

        for col in [
            "headline_probability",
            "logistic_probability",
            "random_forest_probability",
        ]:

            shown[col] = shown[col].map(
                lambda x:
                    f"{x:.2%}"
            )

        shown["income"] = shown[
            "income"
        ].map(
            lambda x:
                f"{x:,.0f}"
        )

        shown["debt_ratio"] = shown[
            "debt_ratio"
        ].map(
            lambda x:
                f"{x:.2f}"
        )

        shown["utilization"] = shown[
            "utilization"
        ].map(
            lambda x:
                f"{x:.2f}"
        )

        shown = shown[
            [
                "timestamp",
                "model",
                "headline_probability",
                "risk_band",
                "logistic_probability",
                "random_forest_probability",
                "age",
                "income",
                "utilization",
                "debt_ratio",
            ]
        ]

        shown.columns = [
            "Time",
            "Headline model",
            "Headline probability",
            "Risk band",
            "Logistic Regression",
            "Random Forest",
            "Age",
            "Income",
            "Utilization",
            "Debt ratio",
        ]

        st.dataframe(
            shown,
            hide_index=True,
            use_container_width=True,
        )

        if len(hdf) > 1:

            ordered = (
                hdf
                .iloc[::-1]
                .reset_index(drop=True)
            )

            ordered.index = (
                ordered.index + 1
            )

            st.markdown(
                "#### Probability across assessments"
            )

            line = ordered[
                [
                    "logistic_probability",
                    "random_forest_probability",
                ]
            ].copy()

            line.columns = [
                LR,
                RF,
            ]

            st.line_chart(
                line,
                height=320,
                color=[
                    COLORS["burgundy_light"],
                    COLORS["gold"],
                ],
            )

        st.markdown(
            "#### Risk band distribution"
        )

        bands = (
            hdf["risk_band"]
            .value_counts()
            .reindex(
                [
                    "Low risk",
                    "Moderate risk",
                    "High risk",
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            bands.to_frame(
                "Assessments"
            ),
            color=COLORS[
                "burgundy_light"
            ],
            height=240,
        )

        st.download_button(
            "Download session history (CSV)",
            hdf
            .to_csv(index=False)
            .encode("utf-8"),
            file_name=(
                "credit_risk_session_history.csv"
            ),
            mime="text/csv",
        )


# =========================================================
# TAB 4 — HOW IT WORKS
# =========================================================

with tab_about:

    section(
        "From inputs to probability",
        "The app turns the applicant profile into the exact "
        "engineered feature structure the stored models expect.",
    )

    steps = [
        (
            "Applicant data",
            "The values entered in the assessment form.",
        ),

        (
            "Validation",
            "Input ranges are enforced and unusual "
            "combinations are flagged.",
        ),

        (
            "Feature engineering",
            "Totals, ratios, log transforms, flags, "
            "interactions, and age groups are derived.",
        ),

        (
            "Feature alignment",
            "The row is reordered to match the stored "
            "training feature names.",
        ),

        (
            "Scaling",
            "Logistic Regression receives features "
            "transformed by the stored scaler.",
        ),

        (
            "Model inference",
            "Both models produce class probabilities "
            "and predicted classes.",
        ),

        (
            "Risk band",
            "The headline probability is mapped to "
            "Low, Moderate, or High.",
        ),

        (
            "Explanation",
            "Logistic coefficients and Random Forest "
            "tree spread describe each estimate.",
        ),

        (
            "Export",
            "The assessment can be downloaded as "
            "CSV or JSON.",
        ),
    ]

    cols = st.columns(3)

    for i, (
        title,
        description,
    ) in enumerate(steps):

        with cols[i % 3]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{i + 1}. {title}**"
                )

                st.caption(
                    description
                )

    # -----------------------------------------------------
    # ENGINEERED VARIABLES
    # -----------------------------------------------------

    section(
        "Engineered variables",
        "Additional variables derived before inference.",
    )

    engineered = pd.DataFrame(
        [
            [
                "total_past_due",
                "30–59 + 60–89 + 90+ past-due events",
            ],

            [
                "severe_delinquency_ratio",
                "90+ day events / (total past due + 1)",
            ],

            [
                "total_credit_lines",
                "Open credit lines + real estate loans",
            ],

            [
                "real_estate_ratio",
                "Real estate loans / (total credit lines + 1)",
            ],

            [
                "log_income",
                "log(1 + monthly income)",
            ],

            [
                "income_per_credit_line",
                "Monthly income / (open lines + 1)",
            ],

            [
                "high_debt_flag",
                "1 when debt ratio > 1",
            ],

            [
                "log_debt_ratio",
                "log(1 + debt ratio)",
            ],

            [
                "utilization_flag",
                "1 when revolving utilization > 1",
            ],

            [
                "age_risk_score",
                "Age × total past-due events",
            ],
        ],
        columns=[
            "Feature",
            "Definition",
        ],
    )

    st.dataframe(
        engineered,
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # LIMITATIONS
    # -----------------------------------------------------

    section(
        "Interpretation and limitations",
        "A model probability is an estimate produced under "
        "the assumptions of the training process. It is not "
        "a guaranteed event or a causal conclusion.",
    )

    html(
        """
        <div class="note">
            <strong>Academic demonstration.</strong>
            This app demonstrates a machine-learning
            credit-risk classification workflow.
            Probabilities and classes are model outputs.
            They are not guaranteed outcomes and must not
            replace real-world financial, lending,
            underwriting, compliance, or credit decisions.
        </div>
        """
    )

    st.write("")

    limitations = [
        (
            "Model uncertainty",
            "Predictions are estimates and can be wrong "
            "for individual cases.",
        ),

        (
            "Data dependence",
            "Performance depends on the quality and "
            "representativeness of the evaluation data.",
        ),

        (
            "No causal reading",
            "Contributions and importances do not "
            "establish cause and effect.",
        ),
    ]

    cols = st.columns(3)

    for col, (
        title,
        description,
    ) in zip(
        cols,
        limitations,
    ):

        with col:

            html(
                f"""
                <div class="profile-card">

                    <div class="profile-label">
                        {title}
                    </div>

                    <div
                        style="
                            margin-top:.4rem;
                            color:#6B6265;
                            font-size:.77rem;
                            line-height:1.55;
                        "
                    >
                        {description}
                    </div>

                </div>
                """
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

html(
    """
    <div
        style="
            text-align:center;
            color:#6B6265;
            font-size:.72rem;
            line-height:1.6;
            padding:.5rem 0 1rem;
        "
    >
        Credit Risk Analytics · Machine learning demonstration
        <br>
        Probabilities shown are model outputs for academic
        exploration only.
    </div>
    """
)


import json
import textwrap
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN TOKENS
# =========================================================

COLORS = {
    "burgundy": "#4A0717",
    "burgundy_light": "#780B27",
    "burgundy_dark": "#32040F",
    "gold": "#E6C866",
    "text": "#271C20",
    "muted": "#6B6265",
    "border": "#E8E2DF",
    "background": "#F7F5F3",
    "green": "#2F9455",
    "green_soft": "#EAF6EE",
    "orange": "#D58A2D",
    "orange_soft": "#FFF3E4",
    "red": "#B32A45",
    "red_soft": "#FCECEF",
    "blue": "#3867A8",
    "blue_soft": "#EDF3FB",
    "white": "#FFFFFF",
}

CSS_VARS = "".join(
    f"--{k.replace('_', '-')}: {v};"
    for k, v in COLORS.items()
)


CSS_RULES = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    color-scheme: light;
}

html, body, [class*="st-"], .stApp {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.main .block-container {
    max-width: 1380px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

/* =====================================================
   GLOBAL TYPOGRAPHY
   ===================================================== */

h1, h2, h3, h4, h5 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--text) !important;
}

[data-testid="stMarkdownContainer"] > p,
[data-testid="stMarkdownContainer"] > ul,
[data-testid="stMarkdownContainer"] > ol,
[data-testid="stMarkdownContainer"] > ul li,
[data-testid="stMarkdownContainer"] > ol li,
[data-testid="stMarkdownContainer"] > h1,
[data-testid="stMarkdownContainer"] > h2,
[data-testid="stMarkdownContainer"] > h3,
[data-testid="stMarkdownContainer"] > h4 {
    color: var(--text) !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--text) !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: var(--burgundy) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {
    color: var(--muted) !important;
}

/* =====================================================
   METRICS
   ===================================================== */

[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--border);
    padding: .75rem .9rem;
    border-radius: 12px;
}

/* =====================================================
   EXPANDERS / ALERTS
   ===================================================== */

[data-testid="stExpander"] details {
    background: #fff;
    border-color: var(--border);
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--text) !important;
}

[data-testid="stAlert"] {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p {
    color: var(--text) !important;
}

/* =====================================================
   TABS
   ===================================================== */

.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p {
    color: var(--muted) !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p {
    color: var(--burgundy) !important;
}

/* =====================================================
   INPUTS
   ===================================================== */

[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="select"] > div {
    background: #fff !important;
}

[data-baseweb="input"] input,
[data-baseweb="select"] *,
[data-testid="stNumberInput"] button {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background: #fff !important;
    color: var(--text) !important;
}

/* =====================================================
   SIDEBAR
   ===================================================== */

[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] :is(
    h1, h2, h3, h4, p, label, li, small
) {
    color: var(--text) !important;
}

[data-testid="stSidebar"]
[data-testid="stCaptionContainer"] * {
    color: var(--muted) !important;
}

/* =====================================================
   CONTAINERS
   ===================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border-radius: 14px;
    border-color: var(--border);
}

/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button,
[data-testid="stDownloadButton"] button {
    width: 100%;
    min-height: 2.6rem;
    border-radius: 10px;
    font-weight: 600;
    background: #fff;
    border: 1px solid var(--border);
}

.stButton > button p,
[data-testid="stDownloadButton"] button p {
    color: var(--text) !important;
}

.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    min-height: 3.05rem;
    border: none;
    background:
        linear-gradient(
            120deg,
            var(--burgundy),
            var(--burgundy-light)
        ) !important;
}

.stButton > button[kind="primary"] p,
[data-testid="stBaseButton-primary"] p {
    color: #fff !important;
}

/* =====================================================
   HERO
   ===================================================== */

.hero {
    padding: 2.3rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1rem;

    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(230, 200, 102, .18),
            transparent 30%
        ),
        linear-gradient(
            120deg,
            var(--burgundy-dark),
            var(--burgundy-light)
        );

    box-shadow:
        0 12px 35px rgba(50, 4, 15, .14);
}

.hero h1 {
    color: #fff !important;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.03;
    margin: 0;
    letter-spacing: -0.025em;
}

.hero p {
    max-width: 790px;
    margin: .85rem 0 0;
    color: rgba(255, 255, 255, .82) !important;
    font-size: .98rem;
    line-height: 1.65;
}

.hero-badge {
    display: inline-block;
    margin-bottom: .8rem;
    padding: .34rem .7rem;
    border: 1px solid rgba(230, 200, 102, .4);
    border-radius: 999px;
    color: var(--gold);
    background: rgba(255, 255, 255, .06);
    font-size: .72rem;
    font-weight: 700;
}

/* =====================================================
   SECTIONS
   ===================================================== */

.section-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1.5rem 0 .15rem;
    color: var(--text);
}

.section-copy {
    color: var(--muted);
    font-size: .86rem;
    line-height: 1.65;
    max-width: 840px;
    margin-bottom: .9rem;
}

/* =====================================================
   KPI CARDS
   ===================================================== */

.kpi {
    height: 100%;
    min-height: 118px;
    padding: 1rem 1.1rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #fff;
    color: var(--text);
    box-shadow:
        0 3px 14px rgba(39, 28, 32, .035);
}

.kpi-label {
    color: var(--muted);
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .03em;
    text-transform: uppercase;
}

.kpi-value {
    margin-top: .25rem;
    color: var(--burgundy);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.1;
}

.kpi-sub {
    margin-top: .4rem;
    color: var(--muted);
    font-size: .74rem;
    line-height: 1.4;
}

/* =====================================================
   RESULT CARD
   ===================================================== */

.result {
    padding: 1.8rem 1.9rem;
    border-radius: 16px;
    color: #fff;

    background:
        radial-gradient(
            circle at 95% 0%,
            rgba(230, 200, 102, .16),
            transparent 28%
        ),
        linear-gradient(
            120deg,
            var(--burgundy-dark),
            var(--burgundy-light)
        );

    box-shadow:
        0 12px 28px rgba(50, 4, 15, .13);
}

.result .small {
    color: var(--gold);
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
}

.result .big {
    color: #fff;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: clamp(2.8rem, 5vw, 4.3rem);
    font-weight: 700;
    line-height: 1;
    margin-top: .25rem;
}

.result .caption {
    color: rgba(255, 255, 255, .8);
    font-size: .85rem;
    margin-top: .55rem;
    max-width: 650px;
    line-height: 1.55;
}

.pill {
    display: inline-block;
    margin-top: .95rem;
    padding: .38rem .82rem;
    border-radius: 999px;
    background: #fff;
    font-weight: 700;
    font-size: .78rem;
}

.track {
    position: relative;
    height: 9px;
    margin-top: 1.55rem;
    border-radius: 99px;
    background:
        linear-gradient(
            90deg,
            var(--green) 0%,
            #D7B64D 50%,
            var(--red) 100%
        );
}

.marker {
    position: absolute;
    top: 50%;
    width: 19px;
    height: 19px;
    border-radius: 50%;
    background: #fff;
    border: 3px solid var(--burgundy-light);
    transform: translate(-50%, -50%);
    box-shadow: 0 2px 8px rgba(0, 0, 0, .35);
}

.ticks {
    display: flex;
    justify-content: space-between;
    margin-top: .48rem;
    font-size: .7rem;
    color: rgba(255, 255, 255, .7);
}

/* =====================================================
   SECOND MODEL CARD
   ===================================================== */

.other {
    padding: 1.3rem 1.35rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #fff;
    height: 100%;
    color: var(--text);
}

.other .k {
    color: var(--muted);
    font-size: .76rem;
    font-weight: 600;
}

.other .v {
    margin-top: .25rem;
    color: var(--burgundy);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
}

.other .b {
    margin-top: .35rem;
    font-size: .79rem;
    font-weight: 700;
}

.other .line {
    height: 6px;
    margin-top: 1.1rem;
    border-radius: 99px;
    background: var(--border);
    overflow: hidden;
}

.other .line-fill {
    height: 100%;
    border-radius: 99px;
}

.other .foot {
    margin-top: .8rem;
    color: var(--muted);
    font-size: .72rem;
    line-height: 1.5;
}

/* =====================================================
   PROFILE CARDS
   ===================================================== */

.profile-card {
    padding: 1rem 1.1rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #fff;
    color: var(--text);
}

.profile-label {
    color: var(--muted);
    font-size: .7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .03em;
}

.profile-value {
    color: var(--text);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: .18rem;
}

/* =====================================================
   FACTORS
   ===================================================== */

.factor {
    display: flex;
    gap: .85rem;
    align-items: flex-start;
    padding: .85rem 1rem;
    margin-bottom: .55rem;
    border: 1px solid var(--border);
    border-radius: 11px;
    background: #fff;
}

.factor-icon {
    width: 30px;
    height: 30px;
    min-width: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-weight: 700;
    font-size: .85rem;
}

.factor-positive {
    background: var(--red-soft);
    color: var(--red);
}

.factor-negative {
    background: var(--green-soft);
    color: var(--green);
}

.factor-neutral {
    background: var(--blue-soft);
    color: var(--blue);
}

.factor-title {
    font-size: .81rem;
    font-weight: 700;
    color: var(--text);
}

.factor-num {
    color: var(--muted);
    font-weight: 500;
    margin-left: .35rem;
}

.factor-text {
    margin-top: .15rem;
    color: var(--muted);
    font-size: .74rem;
    line-height: 1.5;
}

/* =====================================================
   QUALITY
   ===================================================== */

.quality {
    padding: .8rem 1rem;
    border-radius: 11px;
    border: 1px solid var(--border);
    background: #fff;
    color: var(--text);
    font-size: .85rem;
}

.quality-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: .4rem;
    border-radius: 50%;
}

.quality-sub {
    color: var(--muted);
    margin-left: .4rem;
    font-size: .76rem;
}

/* =====================================================
   NOTE
   ===================================================== */

.note {
    padding: .9rem 1rem;
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    background: #FFFDF7;
    color: #4F464A;
    font-size: .82rem;
    line-height: 1.65;
}

/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 700px) {
    .hero {
        padding: 1.5rem 1.2rem;
    }

    .main .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }
}
"""


st.markdown(
    f"<style>:root {{{CSS_VARS}}}{CSS_RULES}</style>",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def html(block: str) -> None:
    """
    Render HTML safely.
    Lines are flattened so Markdown cannot interpret indented HTML as code.
    """
    flat = " ".join(
        line.strip()
        for line in textwrap.dedent(block).splitlines()
        if line.strip()
    )

    st.markdown(flat, unsafe_allow_html=True)


def section(title: str, copy: str = "") -> None:
    html(f'<div class="section-title">{title}</div>')

    if copy:
        html(f'<div class="section-copy">{copy}</div>')


def money(x: float) -> str:
    return f"&#36;{x:,.0f}"


def kpi(
    label_: str,
    value: str,
    sub: str = "",
    color: str | None = None,
) -> str:

    style = f' style="color:{color};"' if color else ""

    return (
        f'<div class="kpi">'
        f'<div class="kpi-label">{label_}</div>'
        f'<div class="kpi-value"{style}>{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def kpi_row(items: list[tuple]) -> None:
    columns = st.columns(len(items))

    for col, item in zip(columns, items):
        with col:
            html(kpi(*item))


def pcard(title: str, value: str) -> str:
    return (
        f'<div class="profile-card">'
        f'<div class="profile-label">{title}</div>'
        f'<div class="profile-value">{value}</div>'
        f'</div>'
    )


def pcard_row(items: list[tuple]) -> None:
    columns = st.columns(len(items))

    for col, (title, value) in zip(columns, items):
        with col:
            html(pcard(title, value))


# =========================================================
# MODEL PACKAGE
# =========================================================

REQUIRED_KEYS = {
    "logistic_model",
    "random_forest_model",
    "scaler",
    "feature_names",
    "comparison",
    "config",
}


@st.cache_resource(show_spinner="Loading trained models...")
def load_package(path: str = "credit_risk_models.pkl") -> dict:
    package = joblib.load(path)

    if not isinstance(package, dict):
        raise TypeError(
            f"{path} must contain a dictionary."
        )

    missing = REQUIRED_KEYS.difference(package.keys())

    if missing:
        raise KeyError(
            f"Model package is missing keys: {sorted(missing)}"
        )

    if not package["feature_names"]:
        raise ValueError(
            "The model package contains no feature_names."
        )

    return package


def normalize_comparison(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        result = data.copy()

    elif isinstance(data, (dict, list)):
        result = pd.DataFrame(data)

    else:
        raise TypeError(
            "Comparison must be a DataFrame, dict, or list."
        )

    if "Model" not in result.columns:
        raise ValueError(
            "Comparison data must contain a 'Model' column."
        )

    return result


try:
    package = load_package()
    comparison = normalize_comparison(
        package["comparison"]
    )

except FileNotFoundError:
    st.error(
        "Model file not found. Place "
        "`credit_risk_models.pkl` next to this app and reload."
    )
    st.stop()

except Exception as exc:
    st.error(
        "The trained model package could not be loaded."
    )

    st.code(
        f"{type(exc).__name__}: {exc}"
    )

    st.stop()


logistic_model = package["logistic_model"]
random_forest_model = package["random_forest_model"]
scaler = package["scaler"]
feature_names = list(package["feature_names"])
config = package["config"]

LR = "Logistic Regression"
RF = "Random Forest"

MODEL_NAMES = [LR, RF]


# =========================================================
# LABELS
# =========================================================

FEATURE_LABELS = {
    "age": "Age",
    "MonthlyIncome": "Monthly income",
    "NumberOfDependents": "Dependents",
    "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
    "DebtRatio": "Debt ratio",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days past due",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days past due",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "MonthlyIncome_missing": "Income missing flag",
    "total_past_due": "Total past-due events",
    "severe_delinquency_ratio": "Severe delinquency ratio",
    "total_credit_lines": "Total credit lines",
    "real_estate_ratio": "Real estate share",
    "log_income": "Income (log)",
    "income_per_credit_line": "Income per credit line",
    "high_debt_flag": "Debt ratio above 1",
    "log_debt_ratio": "Debt ratio (log)",
    "utilization_flag": "Utilization above 100%",
    "age_risk_score": "Age × past-due events",
}


def label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    if name.startswith("age_bin_"):
        return (
            "Age group: "
            + name.replace("age_bin_", "")
        )

    return name


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(inp: dict) -> pd.DataFrame:

    data = pd.DataFrame(
        {
            "age": [inp["age"]],
            "MonthlyIncome": [
                inp["monthly_income"]
            ],
            "NumberOfDependents": [
                inp["dependents"]
            ],
            "RevolvingUtilizationOfUnsecuredLines": [
                inp["utilization"]
            ],
            "DebtRatio": [
                inp["debt_ratio"]
            ],
            "NumberOfTime30-59DaysPastDueNotWorse": [
                inp["late_30"]
            ],
            "NumberOfTime60-89DaysPastDueNotWorse": [
                inp["late_60"]
            ],
            "NumberOfTimes90DaysLate": [
                inp["late_90"]
            ],
            "NumberOfOpenCreditLinesAndLoans": [
                inp["open_lines"]
            ],
            "NumberRealEstateLoansOrLines": [
                inp["real_estate"]
            ],
            "MonthlyIncome_missing": [0],
        }
    )

    d30 = data[
        "NumberOfTime30-59DaysPastDueNotWorse"
    ]

    d60 = data[
        "NumberOfTime60-89DaysPastDueNotWorse"
    ]

    d90 = data[
        "NumberOfTimes90DaysLate"
    ]

    open_lines = data[
        "NumberOfOpenCreditLinesAndLoans"
    ]

    re_lines = data[
        "NumberRealEstateLoansOrLines"
    ]

    data["total_past_due"] = (
        d30 + d60 + d90
    )

    data["severe_delinquency_ratio"] = (
        d90 /
        (data["total_past_due"] + 1)
    )

    data["total_credit_lines"] = (
        open_lines + re_lines
    )

    data["real_estate_ratio"] = (
        re_lines /
        (data["total_credit_lines"] + 1)
    )

    data["log_income"] = np.log1p(
        data["MonthlyIncome"].clip(lower=0)
    )

    data["income_per_credit_line"] = (
        data["MonthlyIncome"] /
        (open_lines + 1)
    )

    data["high_debt_flag"] = (
        data["DebtRatio"] > 1
    ).astype(int)

    data["log_debt_ratio"] = np.log1p(
        data["DebtRatio"].clip(lower=0)
    )

    data["utilization_flag"] = (
        data[
            "RevolvingUtilizationOfUnsecuredLines"
        ] > 1
    ).astype(int)

    data["age_bin"] = pd.cut(
        data["age"],
        bins=[0, 25, 35, 50, 65, 120],
        labels=[
            "young",
            "adult",
            "mid",
            "senior",
            "old",
        ],
        include_lowest=True,
    )

    data["age_risk_score"] = (
        data["age"] *
        data["total_past_due"]
    )

    data = pd.get_dummies(
        data,
        columns=["age_bin"],
        drop_first=True,
        dtype=int,
    )

    data = data.reindex(
        columns=feature_names,
        fill_value=0,
    )

    return (
        data
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def validate_model_inputs(
    X: pd.DataFrame,
) -> None:

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Model expects {len(feature_names)} "
            f"features, got {X.shape[1]}."
        )

    if list(X.columns) != feature_names:
        raise ValueError(
            "Feature order does not match "
            "the training order."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Generated features contain "
            "NaN or infinite values."
        )


# =========================================================
# PREDICTION
# =========================================================

def get_prediction(
    model_name: str,
    X: pd.DataFrame,
) -> tuple[float, int]:

    validate_model_inputs(X)

    if model_name == LR:
        X_model = scaler.transform(X)
        model = logistic_model

    elif model_name == RF:
        X_model = X
        model = random_forest_model

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    probability = float(
        model.predict_proba(X_model)[0, 1]
    )

    prediction = int(
        model.predict(X_model)[0]
    )

    return probability, prediction


# =========================================================
# MODEL EXPLANATION
# =========================================================

def logistic_contributions(
    X: pd.DataFrame,
) -> pd.Series | None:

    try:
        scaled = scaler.transform(X)[0]

        coefs = np.asarray(
            logistic_model.coef_
        )[0]

        return pd.Series(
            coefs * scaled,
            index=feature_names,
        )

    except Exception:
        return None


def rf_tree_stats(
    X: pd.DataFrame,
) -> dict | None:

    try:
        arr = X.to_numpy(dtype=float)

        probabilities = []

        for tree in random_forest_model.estimators_:

            row = tree.predict_proba(arr)[0]

            if len(row) > 1:
                probabilities.append(
                    float(row[1])
                )
            else:
                probabilities.append(0.0)

        probabilities = np.asarray(
            probabilities
        )

        counts, _ = np.histogram(
            probabilities,
            bins=10,
            range=(0, 1),
        )

        hist = pd.DataFrame(
            {"Trees": counts},
            index=[
                f"{i * 10}–{i * 10 + 10}%"
                for i in range(10)
            ],
        )

        return {
            "n": len(probabilities),
            "mean": probabilities.mean(),
            "std": probabilities.std(),
            "p05": np.percentile(
                probabilities,
                5,
            ),
            "p95": np.percentile(
                probabilities,
                95,
            ),
            "votes": int(
                (probabilities >= 0.5).sum()
            ),
            "hist": hist,
        }

    except Exception:
        return None


# =========================================================
# SENSITIVITY
# =========================================================

SENS_VARS = {
    "Revolving utilization": (
        "utilization",
        np.linspace(0, 2, 21),
        float,
    ),

    "Debt ratio": (
        "debt_ratio",
        np.linspace(0, 3, 21),
        float,
    ),

    "Monthly income": (
        "monthly_income",
        np.linspace(500, 15000, 21),
        float,
    ),

    "Age": (
        "age",
        np.arange(18, 91, 4),
        int,
    ),

    "30–59 days past due": (
        "late_30",
        np.arange(0, 11),
        int,
    ),

    "90+ days late": (
        "late_90",
        np.arange(0, 11),
        int,
    ),
}


def sensitivity_table(
    inp: dict,
    name: str,
) -> pd.DataFrame:

    key, grid, cast = SENS_VARS[name]

    rows = []

    for value in grid:

        trial = dict(inp)

        trial[key] = cast(value)

        X = create_features(trial)

        rows.append(
            {
                name: cast(value),
                LR: get_prediction(LR, X)[0],
                RF: get_prediction(RF, X)[0],
            }
        )

    return pd.DataFrame(rows).set_index(name)


# =========================================================
# RISK
# =========================================================

def classify_risk(
    p: float,
) -> tuple[str, str, str]:

    if p < 0.25:
        return (
            "Low risk",
            COLORS["green"],
            "Lower predicted probability of serious delinquency.",
        )

    if p < 0.50:
        return (
            "Moderate risk",
            COLORS["orange"],
            "Intermediate predicted probability of serious delinquency.",
        )

    return (
        "High risk",
        COLORS["red"],
        "Higher predicted probability of serious delinquency.",
    )


def risk_band_short(p: float) -> str:
    return classify_risk(p)[0]


def one_in(p: float) -> str:

    if p <= 0:
        return "—"

    if p < 1:
        return f"1 in {1 / p:,.0f}"

    return "1 in 1"


# =========================================================
# INPUT QUALITY
# =========================================================

def input_warnings(inp: dict) -> list[str]:

    notes = []

    if inp["monthly_income"] == 0:
        notes.append(
            "Monthly income is 0, so income-based "
            "features will be at their minimum."
        )

    if inp["debt_ratio"] > 1:
        notes.append(
            "Debt ratio is above 1: debt exceeds income."
        )

    if inp["utilization"] > 1:
        notes.append(
            "Revolving utilization is above 100%."
        )

    if (
        inp["age"] < 21
        and inp["late_90"] > 0
    ):
        notes.append(
            "Several past-due events at a very "
            "young age is an unusual combination."
        )

    return notes


def profile_quality(
    inp: dict,
) -> tuple[str, str]:

    issues = sum(
        [
            inp["monthly_income"] == 0,
            inp["debt_ratio"] > 5,
            inp["utilization"] > 5,
            inp["age"] < 18 or inp["age"] > 100,
        ]
    )

    if issues == 0:
        return (
            "Inputs valid",
            COLORS["green"],
        )

    if issues <= 2:
        return (
            "Review inputs",
            COLORS["orange"],
        )

    return (
        "Unusual profile",
        COLORS["red"],
    )


# =========================================================
# EXPLANATION
# =========================================================

PAST_DUE = {
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "total_past_due",
}


def explain_feature(
    f: str,
    v: float,
    c: float,
    inp: dict,
) -> str:

    effect = (
        "raises"
        if c > 0
        else "lowers"
    )

    if f == "RevolvingUtilizationOfUnsecuredLines":
        return (
            f"Utilization of "
            f"{inp['utilization']:.0%} "
            f"{effect} the estimated log-odds."
        )

    if f == "DebtRatio":
        return (
            f"A debt ratio of "
            f"{inp['debt_ratio']:.2f} "
            f"{effect} the estimated log-odds."
        )

    if f in PAST_DUE:
        if v > 0:
            return (
                f"{v:.0f} recorded event(s); "
                f"this {effect} the estimated log-odds."
            )

        return (
            "No events recorded for this measure."
        )

    if f in {
        "high_debt_flag",
        "utilization_flag",
    }:

        if v > 0:
            return (
                "Flag is on: the threshold "
                "is exceeded."
            )

        return (
            "Flag is off: the threshold "
            "is not exceeded."
        )

    return (
        f"This feature {effect} the estimated "
        f"log-odds for this applicant."
    )


def build_factor_rows(
    contrib: pd.Series,
    X: pd.DataFrame,
    inp: dict,
    n: int = 8,
) -> list[dict]:

    ranked = (
        contrib.abs()
        .sort_values(ascending=False)
        .head(n)
        .index
    )

    rows = []

    for f in ranked:

        c = float(contrib[f])
        v = float(X.iloc[0][f])

        rows.append(
            {
                "label": label(f),
                "contribution": c,
                "direction": (
                    "positive"
                    if c > 0
                    else "negative"
                    if c < 0
                    else "neutral"
                ),
                "explanation": explain_feature(
                    f,
                    v,
                    c,
                    inp,
                ),
            }
        )

    return rows


def model_params(
    model,
    keys: list[str],
) -> pd.DataFrame:

    try:
        params = model.get_params()

        rows = [
            (k, str(params[k]))
            for k in keys
            if k in params
        ]

    except Exception:
        rows = []

    return pd.DataFrame(
        rows,
        columns=["Parameter", "Value"],
    )


# =========================================================
# STATE / PRESETS
# =========================================================

DEFAULTS = {
    "applicant_age": 41,
    "applicant_monthly_income": 5000.0,
    "applicant_dependents": 1,
    "applicant_revolving_utilization": 0.35,
    "applicant_debt_ratio": 0.40,
    "applicant_open_credit_lines": 8,
    "applicant_real_estate_loans": 1,
    "delinq_30_59": 0,
    "delinq_60_89": 0,
    "delinq_90_plus": 0,
}


PRESETS = {
    "Typical applicant": DEFAULTS,

    "Established, low utilization": {
        "applicant_age": 52,
        "applicant_monthly_income": 9000.0,
        "applicant_dependents": 0,
        "applicant_revolving_utilization": 0.08,
        "applicant_debt_ratio": 0.20,
        "applicant_open_credit_lines": 9,
        "applicant_real_estate_loans": 2,
        "delinq_30_59": 0,
        "delinq_60_89": 0,
        "delinq_90_plus": 0,
    },

    "Stretched, repeated late payments": {
        "applicant_age": 29,
        "applicant_monthly_income": 2200.0,
        "applicant_dependents": 3,
        "applicant_revolving_utilization": 1.40,
        "applicant_debt_ratio": 1.60,
        "applicant_open_credit_lines": 4,
        "applicant_real_estate_loans": 0,
        "delinq_30_59": 3,
        "delinq_60_89": 2,
        "delinq_90_plus": 2,
    },
}


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

for key, value in DEFAULTS.items():
    st.session_state.setdefault(
        key,
        value,
    )

st.session_state.setdefault(
    "assessment_history",
    [],
)

st.session_state.setdefault(
    "result",
    None,
)


# =========================================================
# CALLBACKS
# =========================================================

def apply_preset(name: str) -> None:
    """
    Callback used by preset buttons.

    IMPORTANT:
    Session-state values are changed here before the
    widgets are instantiated during the subsequent rerun.
    """

    preset = PRESETS[name]

    for key, value in preset.items():
        st.session_state[key] = value

    st.session_state["result"] = None


def reset_applicant() -> None:
    """
    Reset applicant values.

    This function is intentionally used ONLY as a button
    callback. It must not be called later in the script
    after widgets have been instantiated.
    """

    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.session_state["result"] = None


def clear_history() -> None:
    st.session_state["assessment_history"] = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("### Model controls")

    model_name = st.selectbox(
        "Headline model",
        MODEL_NAMES,
        key="sidebar_prediction_model",
        help=(
            "Both models are always evaluated. "
            "This selects which model is displayed "
            "as the headline estimate."
        ),
    )

    st.divider()

    st.markdown("### Risk bands")

    st.markdown(
        "🟢 **Low**: below 25%  \n"
        "🟠 **Moderate**: 25% to 50%  \n"
        "🔴 **High**: 50% and above"
    )

    st.divider()

    st.markdown("### Target")

    st.markdown("`SeriousDlqin2yrs`")

    st.caption(
        "Serious delinquency within two years."
    )

    st.divider()

    st.markdown("### Session")

    st.metric(
        "Assessments this session",
        len(
            st.session_state[
                "assessment_history"
            ]
        ),
    )

    if st.session_state["assessment_history"]:

        st.button(
            "Clear assessment history",
            key="clear_history",
            on_click=clear_history,
        )

    st.divider()

    st.caption(
        "Academic demonstration only. "
        "Not for lending, credit approval, "
        "underwriting, or any real financial decision."
    )


# =========================================================
# HERO
# =========================================================

html(
    """
    <div class="hero">
        <div class="hero-badge">
            Machine learning · Credit analytics
        </div>

        <h1>
            Credit Risk Analytics
        </h1>

        <p>
            Explore how two trained models estimate the probability
            of serious delinquency within two years. Compare model
            outputs, inspect engineered features, test what-if
            scenarios, and review how the models were evaluated.
        </p>
    </div>
    """
)


# =========================================================
# TABS
# =========================================================

(
    tab_assess,
    tab_perf,
    tab_history,
    tab_about,
) = st.tabs(
    [
        "Assess an applicant",
        "Model performance",
        "Assessment history",
        "How it works",
    ]
)


# =========================================================
# TAB 1 — ASSESS
# =========================================================

with tab_assess:

    section(
        "Applicant profile",
        "Start with a sample profile or enter your own values. "
        "Derived features are calculated automatically to match "
        "the structure the stored models expect.",
    )

    # -----------------------------------------------------
    # PRESETS
    # -----------------------------------------------------

    preset_cols = st.columns(
        len(PRESETS) + 1
    )

    for col, name in zip(
        preset_cols[:-1],
        PRESETS,
    ):

        col.button(
            name,
            key=f"preset_{name}",
            on_click=apply_preset,
            args=(name,),
        )

    preset_cols[-1].button(
        "Reset",
        key="reset_applicant",
        on_click=reset_applicant,
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(
        3,
        gap="large",
    )

    with c1:

        with st.container(border=True):

            st.markdown(
                "**Personal profile**"
            )

            st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                step=1,
                key="applicant_age",
            )

            st.number_input(
                "Monthly income",
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="applicant_monthly_income",
            )

            st.number_input(
                "Number of dependents",
                min_value=0,
                max_value=20,
                step=1,
                key="applicant_dependents",
            )

    with c2:

        with st.container(border=True):

            st.markdown(
                "**Credit profile**"
            )

            st.number_input(
                "Revolving utilization",
                min_value=0.0,
                max_value=20.0,
                step=0.01,
                format="%.2f",
                key="applicant_revolving_utilization",
                help=(
                    "Balance on revolving credit divided "
                    "by available limits. 1.00 is 100%."
                ),
            )

            st.number_input(
                "Debt ratio",
                min_value=0.0,
                max_value=20.0,
                step=0.01,
                format="%.2f",
                key="applicant_debt_ratio",
                help=(
                    "Monthly debt payments divided "
                    "by monthly income."
                ),
            )

            st.number_input(
                "Open credit lines and loans",
                min_value=0,
                max_value=100,
                step=1,
                key="applicant_open_credit_lines",
            )

    with c3:

        with st.container(border=True):

            st.markdown(
                "**Payment history**"
            )

            st.number_input(
                "30–59 days past due",
                min_value=0,
                max_value=50,
                step=1,
                key="delinq_30_59",
            )

            st.number_input(
                "60–89 days past due",
                min_value=0,
                max_value=50,
                step=1,
                key="delinq_60_89",
            )

            st.number_input(
                "90+ days late",
                min_value=0,
                max_value=50,
                step=1,
                key="delinq_90_plus",
            )

    with st.container(border=True):

        st.markdown(
            "**Property profile**"
        )

        st.number_input(
            "Real estate loans and lines",
            min_value=0,
            max_value=50,
            step=1,
            key="applicant_real_estate_loans",
        )

    # -----------------------------------------------------
    # CURRENT INPUTS
    # -----------------------------------------------------

    current_inputs = {
        "age": st.session_state[
            "applicant_age"
        ],

        "monthly_income": st.session_state[
            "applicant_monthly_income"
        ],

        "dependents": st.session_state[
            "applicant_dependents"
        ],

        "utilization": st.session_state[
            "applicant_revolving_utilization"
        ],

        "debt_ratio": st.session_state[
            "applicant_debt_ratio"
        ],

        "open_lines": st.session_state[
            "applicant_open_credit_lines"
        ],

        "real_estate": st.session_state[
            "applicant_real_estate_loans"
        ],

        "late_30": st.session_state[
            "delinq_30_59"
        ],

        "late_60": st.session_state[
            "delinq_60_89"
        ],

        "late_90": st.session_state[
            "delinq_90_plus"
        ],
    }

    # -----------------------------------------------------
    # INPUT QUALITY
    # -----------------------------------------------------

    quality_text, quality_color = (
        profile_quality(current_inputs)
    )

    html(
        f"""
        <div class="quality">
            <span
                class="quality-dot"
                style="background:{quality_color};"
            ></span>

            <strong>{quality_text}</strong>

            <span class="quality-sub">
                All fields are within the permitted ranges.
            </span>
        </div>
        """
    )

    for note in input_warnings(
        current_inputs
    ):
        st.warning(
            note,
            icon="⚠️",
        )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    _, mid, _ = st.columns(
        [1, 1.3, 1]
    )

    with mid:

        predict_clicked = st.button(
            "Estimate credit risk",
            key="predict_credit_risk",
            type="primary",
        )

    if predict_clicked:

        try:

            X_input = create_features(
                current_inputs
            )

            preds = {
                name: get_prediction(
                    name,
                    X_input,
                )
                for name in MODEL_NAMES
            }

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            st.session_state["result"] = {
                "inputs": dict(
                    current_inputs
                ),
                "features": X_input.copy(),
                "preds": preds,
                "timestamp": timestamp,
            }

            headline_probability = preds[
                model_name
            ][0]

            st.session_state[
                "assessment_history"
            ].insert(
                0,
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "headline_probability":
                        headline_probability,
                    "logistic_probability":
                        preds[LR][0],
                    "random_forest_probability":
                        preds[RF][0],
                    "risk_band":
                        risk_band_short(
                            headline_probability
                        ),
                    "age":
                        current_inputs["age"],
                    "income":
                        current_inputs[
                            "monthly_income"
                        ],
                    "debt_ratio":
                        current_inputs[
                            "debt_ratio"
                        ],
                    "utilization":
                        current_inputs[
                            "utilization"
                        ],
                },
            )

            st.session_state[
                "assessment_history"
            ] = st.session_state[
                "assessment_history"
            ][:25]

        except Exception as exc:

            st.error(
                "The prediction could not be generated."
            )

            st.code(
                f"{type(exc).__name__}: {exc}"
            )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    result = st.session_state.get(
        "result"
    )

    if not result:

        html(
            """
            <div class="note">
                <strong>No estimate yet.</strong>
                Choose a sample profile or enter applicant
                values, then select
                <strong>Estimate credit risk</strong>.
            </div>
            """
        )

    else:

        ri = result["inputs"]
        RX = result["features"]

        section(
            "Assessment result",
            "The headline result follows the model chosen "
            "in the sidebar. Both models are always evaluated "
            "so you can compare them.",
        )

        if ri != current_inputs:

            st.info(
                "Inputs have changed since this estimate. "
                "Select “Estimate credit risk” to update it."
            )

        probability, prediction = (
            result["preds"][model_name]
        )

        (
            risk_label,
            risk_color,
            risk_description,
        ) = classify_risk(
            probability
        )

        other_name = next(
            name
            for name in MODEL_NAMES
            if name != model_name
        )

        other_prob, _ = (
            result["preds"][other_name]
        )

        (
            other_label,
            other_color,
            _,
        ) = classify_risk(
            other_prob
        )

        gap = abs(
            probability - other_prob
        )

        same_band = (
            risk_label == other_label
        )

        p_lr = result["preds"][LR][0]
        p_rf = result["preds"][RF][0]

        avg_p = (
            p_lr + p_rf
        ) / 2

        (
            avg_label,
            avg_color,
            _,
        ) = classify_risk(avg_p)

        # -------------------------------------------------
        # KPI ROW
        # -------------------------------------------------

        kpi_row(
            [
                (
                    "Headline probability",
                    f"{probability:.1%}",
                    model_name,
                ),

                (
                    "Risk band",
                    risk_label,
                    "Based on the app's thresholds",
                    risk_color,
                ),

                (
                    "Model gap",
                    f"{gap:.1%}",
                    "Absolute probability difference",
                ),

                (
                    "Model agreement",
                    (
                        "Same band"
                        if same_band
                        else "Different bands"
                    ),
                    f"{model_name} vs {other_name}",
                    (
                        COLORS["green"]
                        if same_band
                        else COLORS["orange"]
                    ),
                ),
            ]
        )

        st.write("")

        # -------------------------------------------------
        # ADDITIONAL METRICS
        # -------------------------------------------------

        kpi_row(
            [
                (
                    "Ensemble average",
                    f"{avg_p:.1%}",
                    f"Mean of both models · {avg_label}",
                    avg_color,
                ),

                (
                    "Per 100 similar",
                    f"{probability * 100:.1f}",
                    "Reading if perfectly calibrated",
                ),

                (
                    "Odds",
                    one_in(probability),
                    "Approximate one-in-N frequency",
                ),

                (
                    "Distance from 50%",
                    f"{(probability - 0.5) * 100:+.1f} pts",
                    "Negative means below the high-risk line",
                ),
            ]
        )

        st.write("")

        # -------------------------------------------------
        # HEADLINE RESULT
        # -------------------------------------------------

        left, right = st.columns(
            [2, 1],
            gap="medium",
        )

        with left:

            marker = min(
                max(probability * 100, 2),
                98,
            )

            html(
                f"""
                <div class="result">

                    <div class="small">
                        Predicted probability · {model_name}
                    </div>

                    <div class="big">
                        {probability:.1%}
                    </div>

                    <div class="caption">
                        {risk_description}
                    </div>

                    <div
                        class="pill"
                        style="color:{risk_color};"
                    >
                        {risk_label}
                    </div>

                    <div class="track">
                        <div
                            class="marker"
                            style="left:{marker}%;"
                        ></div>
                    </div>

                    <div class="ticks">
                        <span>0%</span>
                        <span>25%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>

                </div>
                """
            )

        with right:

            width = min(
                max(other_prob * 100, 2),
                98,
            )

            html(
                f"""
                <div class="other">

                    <div class="k">
                        Second opinion · {other_name}
                    </div>

                    <div class="v">
                        {other_prob:.1%}
                    </div>

                    <div
                        class="b"
                        style="color:{other_color};"
                    >
                        {other_label}
                    </div>

                    <div class="line">
                        <div
                            class="line-fill"
                            style="
                                width:{width}%;
                                background:{other_color};
                            "
                        ></div>
                    </div>

                    <div class="foot">
                        Independent output for the same
                        engineered feature row.
                    </div>

                </div>
                """
            )

        if same_band:

            st.success(
                f"Both models place this applicant "
                f"in the same band: "
                f"{risk_label.lower()}.",
                icon="✅",
            )

        else:

            st.warning(
                f"The models disagree: "
                f"{model_name} says "
                f"{risk_label.lower()}, "
                f"{other_name} says "
                f"{other_label.lower()}. "
                f"The gap is {gap:.1%}.",
                icon="⚖️",
            )

        # -------------------------------------------------
        # SNAPSHOT
        # -------------------------------------------------

        section(
            "Applicant snapshot",
            "The profile used for this estimate, "
            "plus values derived from it.",
        )

        total_late = (
            ri["late_30"]
            + ri["late_60"]
            + ri["late_90"]
        )

        pcard_row(
            [
                (
                    "Age",
                    f"{ri['age']:.0f}",
                ),

                (
                    "Monthly income",
                    money(
                        ri["monthly_income"]
                    ),
                ),

                (
                    "Utilization",
                    f"{ri['utilization']:.2f}",
                ),

                (
                    "Debt ratio",
                    f"{ri['debt_ratio']:.2f}",
                ),

                (
                    "Past-due events",
                    f"{total_late:.0f}",
                ),
            ]
        )

        st.write("")

        pcard_row(
            [
                (
                    "Est. monthly debt",
                    money(
                        ri["debt_ratio"]
                        * ri["monthly_income"]
                    ),
                ),

                (
                    "Income per credit line",
                    money(
                        ri["monthly_income"]
                        / (
                            ri["open_lines"]
                            + 1
                        )
                    ),
                ),

                (
                    "Total credit lines",
                    f"{ri['open_lines'] + ri['real_estate']:.0f}",
                ),

                (
                    "Severe late share",
                    f"{ri['late_90'] / (total_late + 1):.2f}",
                ),

                (
                    "Dependents",
                    f"{ri['dependents']:.0f}",
                ),
            ]
        )

        # -------------------------------------------------
        # DIAGNOSTICS
        # -------------------------------------------------

        contrib = logistic_contributions(
            RX
        )

        tree = rf_tree_stats(
            RX
        )

        section(
            "Model diagnostics",
            "How each model arrived at its number. "
            "The Logistic Regression view is exact for "
            "this row; the Random Forest view shows how "
            "much its individual trees disagree.",
        )

        dl, dr = st.columns(
            2,
            gap="large",
        )

        with dl:

            with st.container(
                border=True
            ):

                st.markdown(
                    "**Logistic Regression**"
                )

                if contrib is not None:

                    try:
                        intercept = float(
                            np.asarray(
                                logistic_model.intercept_
                            )[0]
                        )

                    except Exception:
                        intercept = 0.0

                    total = float(
                        contrib.sum()
                    )

                    logit = (
                        intercept + total
                    )

                    up = int(
                        (contrib > 0).sum()
                    )

                    down = int(
                        (contrib < 0).sum()
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Intercept",
                        f"{intercept:+.3f}",
                    )

                    b.metric(
                        "Sum of contributions",
                        f"{total:+.3f}",
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Total log-odds",
                        f"{logit:+.3f}",
                    )

                    b.metric(
                        "Drivers up / down",
                        f"{up} / {down}",
                    )

                    calculated_probability = (
                        1 /
                        (
                            1 +
                            np.exp(-logit)
                        )
                    )

                    st.caption(
                        "Probability = "
                        "1 / (1 + e^−log-odds) = "
                        f"{calculated_probability:.2%}. "
                        f"Model output: {p_lr:.2%}."
                    )

                else:

                    st.info(
                        "Contributions are unavailable "
                        "for the stored Logistic Regression model."
                    )

        with dr:

            with st.container(
                border=True
            ):

                st.markdown(
                    "**Random Forest**"
                )

                if tree is not None:

                    a, b = st.columns(2)

                    a.metric(
                        "Mean tree probability",
                        f"{tree['mean']:.1%}",
                    )

                    b.metric(
                        "Tree disagreement (std)",
                        f"{tree['std']:.3f}",
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Trees voting delinquent",
                        f"{tree['votes']} / {tree['n']}",
                    )

                    b.metric(
                        "Middle 90% of trees",
                        f"{tree['p05']:.0%} – {tree['p95']:.0%}",
                    )

                    st.bar_chart(
                        tree["hist"],
                        height=170,
                        color=COLORS[
                            "burgundy_light"
                        ],
                    )

                    st.caption(
                        "Distribution of individual tree "
                        "probabilities for this applicant."
                    )

                else:

                    st.info(
                        "Per-tree statistics are unavailable "
                        "for the stored Random Forest model."
                    )

        # -------------------------------------------------
        # DRIVERS
        # -------------------------------------------------

        if contrib is not None:

            section(
                "What influenced the Logistic Regression estimate",
                "Each value is a feature's push on the log-odds "
                "for this applicant. Positive values raise the "
                "estimate, negative values lower it. These are "
                "model explanations, not causal effects.",
            )

            factors = build_factor_rows(
                contrib,
                RX,
                ri,
                n=8,
            )

            fl, fr = st.columns(
                [1.15, 1],
                gap="large",
            )

            with fl:

                for fac in factors:

                    icon, css = {
                        "positive": (
                            "↑",
                            "factor-positive",
                        ),

                        "negative": (
                            "↓",
                            "factor-negative",
                        ),

                        "neutral": (
                            "•",
                            "factor-neutral",
                        ),
                    }.get(
                        fac["direction"],
                        (
                            "•",
                            "factor-neutral",
                        ),
                    )

                    html(
                        f"""
                        <div class="factor">

                            <div
                                class="factor-icon {css}"
                            >
                                {icon}
                            </div>

                            <div>

                                <div class="factor-title">
                                    {fac["label"]}

                                    <span
                                        class="factor-num"
                                    >
                                        {fac["contribution"]:+.3f}
                                    </span>
                                </div>

                                <div class="factor-text">
                                    {fac["explanation"]}
                                </div>

                            </div>

                        </div>
                        """
                    )

            with fr:

                chart = pd.DataFrame(
                    {
                        "Effect on risk": [
                            f["contribution"]
                            for f in factors
                        ]
                    },
                    index=[
                        f["label"]
                        for f in factors
                    ],
                ).sort_values(
                    "Effect on risk"
                )

                st.bar_chart(
                    chart,
                    horizontal=True,
                    color=COLORS[
                        "burgundy_light"
                    ],
                )

                st.caption(
                    "Log-odds contribution of the "
                    "eight strongest drivers."
                )

        # -------------------------------------------------
        # WHAT-IF
        # -------------------------------------------------

        section(
            "What-if analysis",
            "Change one input while holding the rest "
            "of this applicant's profile fixed, and see "
            "how both models respond.",
        )

        sens_name = st.selectbox(
            "Input to vary",
            list(SENS_VARS),
            key="sens_var",
        )

        try:

            sens = sensitivity_table(
                ri,
                sens_name,
            )

            st.line_chart(
                sens,
                height=320,
                color=[
                    COLORS["burgundy_light"],
                    COLORS["gold"],
                ],
            )

            lo = sens.iloc[0]
            hi = sens.iloc[-1]

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                f"{LR} at lowest value",
                f"{lo[LR]:.1%}",
            )

            s2.metric(
                f"{LR} at highest value",
                f"{hi[LR]:.1%}",
            )

            s3.metric(
                f"{RF} at lowest value",
                f"{lo[RF]:.1%}",
            )

            s4.metric(
                f"{RF} at highest value",
                f"{hi[RF]:.1%}",
            )

            st.caption(
                f"Range shown: "
                f"{sens.index[0]:,.2f} to "
                f"{sens.index[-1]:,.2f}. "
                "Derived features are recalculated "
                "at every point."
            )

        except Exception as exc:

            st.warning(
                "What-if analysis could not be computed: "
                f"{exc}"
            )

        # -------------------------------------------------
        # MODEL COMPARISON
        # -------------------------------------------------

        section(
            "Model comparison for this applicant",
            "Both models receive the same engineered row. "
            "Logistic Regression uses the stored scaler; "
            "Random Forest uses the raw engineered values.",
        )

        rows = []

        for name in MODEL_NAMES:

            p, pred = result[
                "preds"
            ][name]

            rows.append(
                {
                    "Model": name,
                    "Probability": f"{p:.2%}",
                    "Risk band":
                        risk_band_short(p),
                    "Predicted class":
                        "1" if pred else "0",
                    "Odds": one_in(p),
                }
            )

        rows.append(
            {
                "Model": "Ensemble average",
                "Probability": f"{avg_p:.2%}",
                "Risk band": avg_label,
                "Predicted class":
                    "1"
                    if avg_p >= 0.5
                    else "0",
                "Odds": one_in(avg_p),
            }
        )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "View engineered feature values"
        ):

            disp = RX.T.copy()

            disp.columns = ["Value"]

            disp.index = [
                label(feature)
                for feature in disp.index
            ]

            disp.index.name = "Feature"

            st.dataframe(
                disp,
                height=380,
                use_container_width=True,
            )

        with st.expander(
            "Prediction notes and limitations"
        ):

            st.markdown(
                "- The probability is a **model output**, "
                "not a guaranteed outcome.\n"
                "- Risk bands are presentation thresholds "
                "defined by this application.\n"
                "- Logistic Regression contributions describe "
                "model behaviour for this row, not causal "
                "relationships.\n"
                "- What-if curves show model sensitivity, "
                "not what would happen to a real applicant.\n"
                "- Random Forest tree spread reflects model "
                "variance, not calibrated uncertainty.\n"
                "- This app must not be used for real lending, "
                "underwriting, or credit decisions."
            )

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------

        section(
            "Export assessment",
            "Download this assessment for reporting "
            "or further analysis.",
        )

        report = pd.DataFrame(
            [
                {
                    **ri,

                    **{
                        f"{name} probability": p
                        for name, (p, _)
                        in result["preds"].items()
                    },

                    **{
                        f"{name} class": c
                        for name, (_, c)
                        in result["preds"].items()
                    },

                    "ensemble_probability":
                        avg_p,

                    "headline_model":
                        model_name,

                    "headline_risk_band":
                        risk_label,

                    "headline_probability":
                        probability,

                    "timestamp":
                        result["timestamp"],
                }
            ]
        )

        report_json = json.dumps(
            {
                "timestamp":
                    result["timestamp"],

                "headline_model":
                    model_name,

                "headline_probability":
                    probability,

                "headline_risk_band":
                    risk_label,

                "ensemble_probability":
                    avg_p,

                "inputs":
                    ri,

                "predictions": {
                    name: {
                        "probability": p,
                        "class": c,
                        "risk_band":
                            risk_band_short(p),
                    }

                    for name, (p, c)
                    in result["preds"].items()
                },
            },
            indent=2,
            default=str,
        )

        e1, e2, e3 = st.columns(3)

        with e1:

            st.download_button(
                "Download CSV",
                report
                .to_csv(index=False)
                .encode("utf-8"),
                file_name=(
                    "credit_risk_assessment.csv"
                ),
                mime="text/csv",
            )

        with e2:

            st.download_button(
                "Download JSON",
                report_json.encode("utf-8"),
                file_name=(
                    "credit_risk_assessment.json"
                ),
                mime="application/json",
            )

        with e3:

            # =================================================
            # CRITICAL FIX
            #
            # DO NOT:
            #
            # if st.button(...):
            #     reset_applicant()
            #     st.rerun()
            #
            # The callback runs before the next script
            # execution, so applicant widgets can safely
            # receive the reset values.
            # =================================================

            st.button(
                "Start new assessment",
                key="new_assessment",
                on_click=reset_applicant,
            )


# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

with tab_perf:

    section(
        "Model performance",
        "Metrics stored with the project package, measured "
        "on the evaluation data used during model development. "
        "They describe recorded results, not future performance.",
    )

    METRIC_COLUMNS = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "roc_auc",
        "auc",
    }

    numeric = comparison.copy()

    metric_cols = [
        col
        for col in numeric.columns
        if col.lower() in METRIC_COLUMNS
    ]

    for col in metric_cols:
        numeric[col] = pd.to_numeric(
            numeric[col],
            errors="coerce",
        )

    if metric_cols:

        # -------------------------------------------------
        # SCORECARD
        # -------------------------------------------------

        st.markdown(
            "#### Model evaluation metrics"
        )

        cards = []

        for col in metric_cols:

            series = (
                numeric
                .set_index("Model")[col]
                .dropna()
            )

            if series.empty:
                continue

            best_model = series.idxmax()
            best_value = series.max()

            cards.append(
                (
                    col.replace(
                        "_",
                        " "
                    ).title(),

                    f"{best_value:.2%}",

                    f"{best_model} · "
                    f"recorded evaluation value",
                )
            )

        for i in range(
            0,
            len(cards),
            3,
        ):

            kpi_row(
                cards[i:i + 3]
            )

            st.write("")

        # -------------------------------------------------
        # MODEL TABLE
        # -------------------------------------------------

        st.markdown(
            "#### Scorecard by model"
        )

        for _, row in numeric.iterrows():

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{row['Model']}**"
                )

                vals = [
                    (metric, row[metric])
                    for metric in metric_cols
                    if pd.notna(row[metric])
                ]

                for j in range(
                    0,
                    len(vals),
                    3,
                ):

                    cols = st.columns(3)

                    for cc, (metric, value) in zip(
                        cols,
                        vals[j:j + 3],
                    ):

                        cc.metric(
                            metric.replace(
                                "_",
                                " "
                            ).title(),
                            f"{value:.2%}",
                        )

        # -------------------------------------------------
        # SPREAD TABLE
        # -------------------------------------------------

        spread_rows = []

        for col in metric_cols:

            series = (
                numeric
                .set_index("Model")[col]
                .dropna()
            )

            if len(series) > 1:

                spread_rows.append(
                    {
                        "Metric": col,

                        "Recorded highest":
                            f"{series.max():.2%}",

                        "Recorded lowest":
                            f"{series.min():.2%}",

                        "Difference (pts)":
                            f"{(
                                series.max()
                                - series.min()
                            ) * 100:.2f}",
                    }
                )

        if spread_rows:

            st.markdown(
                "#### Metric differences"
            )

            st.dataframe(
                pd.DataFrame(
                    spread_rows
                ),
                hide_index=True,
                use_container_width=True,
            )

    # -----------------------------------------------------
    # FULL TABLE
    # -----------------------------------------------------

    st.markdown(
        "#### Full evaluation table"
    )

    display = numeric.copy()

    for col in metric_cols:

        display[col] = numeric[col].map(
            lambda x:
                f"{x:.2%}"
                if pd.notna(x)
                else "N/A"
        )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
    )

    if metric_cols:

        chartable = (
            numeric
            .set_index("Model")[metric_cols]
            .dropna(how="all")
        )

        if not chartable.empty:

            st.markdown(
                "#### Metric comparison"
            )

            st.bar_chart(
                chartable.T,
                height=360,
            )

            st.caption(
                "Stored evaluation metrics per model. "
                "This does not reflect deployment cost, "
                "fairness, calibration, or population change."
            )

    # -----------------------------------------------------
    # MODEL DETAILS
    # -----------------------------------------------------

    section(
        "Model details",
        "Configuration of the two stored models.",
    )

    ml, mr = st.columns(
        2,
        gap="large",
    )

    with ml:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Logistic Regression**"
            )

            st.dataframe(
                model_params(
                    logistic_model,
                    [
                        "C",
                        "penalty",
                        "solver",
                        "max_iter",
                        "class_weight",
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )

            try:

                st.metric(
                    "Coefficients",
                    int(
                        np.asarray(
                            logistic_model.coef_
                        ).shape[1]
                    ),
                )

            except Exception:
                pass

    with mr:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Random Forest**"
            )

            st.dataframe(
                model_params(
                    random_forest_model,
                    [
                        "n_estimators",
                        "max_depth",
                        "min_samples_leaf",
                        "min_samples_split",
                        "max_features",
                        "class_weight",
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )

            try:

                depths = [
                    tree.get_depth()
                    for tree
                    in random_forest_model.estimators_
                ]

                a, b = st.columns(2)

                a.metric(
                    "Average tree depth",
                    f"{np.mean(depths):.1f}",
                )

                b.metric(
                    "Deepest tree",
                    int(np.max(depths)),
                )

            except Exception:
                pass

    # -----------------------------------------------------
    # LOGISTIC COEFFICIENTS
    # -----------------------------------------------------

    try:

        coefs = pd.Series(
            np.asarray(
                logistic_model.coef_
            )[0],
            index=[
                label(feature)
                for feature
                in feature_names
            ],
        )

        section(
            "Logistic Regression coefficients",
            "Coefficients on scaled features, so their sizes "
            "are comparable. Positive values raise the estimated "
            "log-odds of delinquency.",
        )

        top = (
            coefs
            .reindex(
                coefs.abs()
                .sort_values(
                    ascending=False
                )
                .head(15)
                .index
            )
            .sort_values()
        )

        st.bar_chart(
            top.to_frame(
                "Coefficient"
            ),
            horizontal=True,
            color=COLORS[
                "burgundy_light"
            ],
        )

    except Exception:
        pass

    # -----------------------------------------------------
    # RANDOM FOREST IMPORTANCE
    # -----------------------------------------------------

    section(
        "Random Forest feature importance",
        "How much each feature contributes to the forest's "
        "split-based impurity reduction. It is not a causal "
        "effect and does not explain an individual applicant.",
    )

    try:

        importances = (
            random_forest_model
            .feature_importances_
        )

        if len(importances) == len(
            feature_names
        ):

            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": [
                            label(feature)
                            for feature
                            in feature_names
                        ],

                        "Importance":
                            importances,
                    }
                )
                .sort_values(
                    "Importance",
                    ascending=False,
                )
            )

            st.bar_chart(
                importance_df
                .head(15)
                .set_index("Feature")
                .sort_values(
                    "Importance"
                ),
                horizontal=True,
                color=COLORS[
                    "burgundy_light"
                ],
            )

            with st.expander(
                "Top 10 features as a table"
            ):

                top10 = (
                    importance_df
                    .head(10)
                    .copy()
                )

                top10["Importance"] = (
                    top10["Importance"]
                    .map(
                        lambda x:
                            f"{x:.4f}"
                    )
                )

                st.dataframe(
                    top10,
                    hide_index=True,
                    use_container_width=True,
                )

        else:

            st.warning(
                "Feature importance length does not "
                "match feature_names."
            )

    except AttributeError:

        st.info(
            "The stored Random Forest model does not "
            "expose feature_importances_."
        )

    except Exception as exc:

        st.warning(
            "Could not display feature importance: "
            f"{exc}"
        )

    with st.expander(
        "Stored project configuration"
    ):

        st.write(config)


# =========================================================
# TAB 3 — HISTORY
# =========================================================

with tab_history:

    section(
        "Assessment history",
        "Estimates generated in this browser session. "
        "They live in Streamlit session state and are "
        "not saved by the app.",
    )

    history = st.session_state[
        "assessment_history"
    ]

    if not history:

        html(
            """
            <div class="note">
                No assessments yet. Generate an estimate
                on the first tab to see it here.
            </div>
            """
        )

    else:

        hdf = pd.DataFrame(
            history
        )

        h1, h2, h3, h4 = st.columns(4)

        h1.metric(
            "Assessments",
            len(hdf),
        )

        h2.metric(
            "Average probability",
            f"{hdf['headline_probability'].mean():.1%}",
        )

        h3.metric(
            "Highest probability",
            f"{hdf['headline_probability'].max():.1%}",
        )

        h4.metric(
            "High-risk share",
            f"{(
                hdf['risk_band'] == 'High risk'
            ).mean():.0%}",
        )

        shown = hdf.copy()

        for col in [
            "headline_probability",
            "logistic_probability",
            "random_forest_probability",
        ]:

            shown[col] = shown[col].map(
                lambda x:
                    f"{x:.2%}"
            )

        shown["income"] = shown[
            "income"
        ].map(
            lambda x:
                f"{x:,.0f}"
        )

        shown["debt_ratio"] = shown[
            "debt_ratio"
        ].map(
            lambda x:
                f"{x:.2f}"
        )

        shown["utilization"] = shown[
            "utilization"
        ].map(
            lambda x:
                f"{x:.2f}"
        )

        shown = shown[
            [
                "timestamp",
                "model",
                "headline_probability",
                "risk_band",
                "logistic_probability",
                "random_forest_probability",
                "age",
                "income",
                "utilization",
                "debt_ratio",
            ]
        ]

        shown.columns = [
            "Time",
            "Headline model",
            "Headline probability",
            "Risk band",
            "Logistic Regression",
            "Random Forest",
            "Age",
            "Income",
            "Utilization",
            "Debt ratio",
        ]

        st.dataframe(
            shown,
            hide_index=True,
            use_container_width=True,
        )

        if len(hdf) > 1:

            ordered = (
                hdf
                .iloc[::-1]
                .reset_index(drop=True)
            )

            ordered.index = (
                ordered.index + 1
            )

            st.markdown(
                "#### Probability across assessments"
            )

            line = ordered[
                [
                    "logistic_probability",
                    "random_forest_probability",
                ]
            ].copy()

            line.columns = [
                LR,
                RF,
            ]

            st.line_chart(
                line,
                height=320,
                color=[
                    COLORS["burgundy_light"],
                    COLORS["gold"],
                ],
            )

        st.markdown(
            "#### Risk band distribution"
        )

        bands = (
            hdf["risk_band"]
            .value_counts()
            .reindex(
                [
                    "Low risk",
                    "Moderate risk",
                    "High risk",
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            bands.to_frame(
                "Assessments"
            ),
            color=COLORS[
                "burgundy_light"
            ],
            height=240,
        )

        st.download_button(
            "Download session history (CSV)",
            hdf
            .to_csv(index=False)
            .encode("utf-8"),
            file_name=(
                "credit_risk_session_history.csv"
            ),
            mime="text/csv",
        )


# =========================================================
# TAB 4 — HOW IT WORKS
# =========================================================

with tab_about:

    section(
        "From inputs to probability",
        "The app turns the applicant profile into the exact "
        "engineered feature structure the stored models expect.",
    )

    steps = [
        (
            "Applicant data",
            "The values entered in the assessment form.",
        ),

        (
            "Validation",
            "Input ranges are enforced and unusual "
            "combinations are flagged.",
        ),

        (
            "Feature engineering",
            "Totals, ratios, log transforms, flags, "
            "interactions, and age groups are derived.",
        ),

        (
            "Feature alignment",
            "The row is reordered to match the stored "
            "training feature names.",
        ),

        (
            "Scaling",
            "Logistic Regression receives features "
            "transformed by the stored scaler.",
        ),

        (
            "Model inference",
            "Both models produce class probabilities "
            "and predicted classes.",
        ),

        (
            "Risk band",
            "The headline probability is mapped to "
            "Low, Moderate, or High.",
        ),

        (
            "Explanation",
            "Logistic coefficients and Random Forest "
            "tree spread describe each estimate.",
        ),

        (
            "Export",
            "The assessment can be downloaded as "
            "CSV or JSON.",
        ),
    ]

    cols = st.columns(3)

    for i, (
        title,
        description,
    ) in enumerate(steps):

        with cols[i % 3]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{i + 1}. {title}**"
                )

                st.caption(
                    description
                )

    # -----------------------------------------------------
    # ENGINEERED VARIABLES
    # -----------------------------------------------------

    section(
        "Engineered variables",
        "Additional variables derived before inference.",
    )

    engineered = pd.DataFrame(
        [
            [
                "total_past_due",
                "30–59 + 60–89 + 90+ past-due events",
            ],

            [
                "severe_delinquency_ratio",
                "90+ day events / (total past due + 1)",
            ],

            [
                "total_credit_lines",
                "Open credit lines + real estate loans",
            ],

            [
                "real_estate_ratio",
                "Real estate loans / (total credit lines + 1)",
            ],

            [
                "log_income",
                "log(1 + monthly income)",
            ],

            [
                "income_per_credit_line",
                "Monthly income / (open lines + 1)",
            ],

            [
                "high_debt_flag",
                "1 when debt ratio > 1",
            ],

            [
                "log_debt_ratio",
                "log(1 + debt ratio)",
            ],

            [
                "utilization_flag",
                "1 when revolving utilization > 1",
            ],

            [
                "age_risk_score",
                "Age × total past-due events",
            ],
        ],
        columns=[
            "Feature",
            "Definition",
        ],
    )

    st.dataframe(
        engineered,
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # LIMITATIONS
    # -----------------------------------------------------

    section(
        "Interpretation and limitations",
        "A model probability is an estimate produced under "
        "the assumptions of the training process. It is not "
        "a guaranteed event or a causal conclusion.",
    )

    html(
        """
        <div class="note">
            <strong>Academic demonstration.</strong>
            This app demonstrates a machine-learning
            credit-risk classification workflow.
            Probabilities and classes are model outputs.
            They are not guaranteed outcomes and must not
            replace real-world financial, lending,
            underwriting, compliance, or credit decisions.
        </div>
        """
    )

    st.write("")

    limitations = [
        (
            "Model uncertainty",
            "Predictions are estimates and can be wrong "
            "for individual cases.",
        ),

        (
            "Data dependence",
            "Performance depends on the quality and "
            "representativeness of the evaluation data.",
        ),

        (
            "No causal reading",
            "Contributions and importances do not "
            "establish cause and effect.",
        ),
    ]

    cols = st.columns(3)

    for col, (
        title,
        description,
    ) in zip(
        cols,
        limitations,
    ):

        with col:

            html(
                f"""
                <div class="profile-card">

                    <div class="profile-label">
                        {title}
                    </div>

                    <div
                        style="
                            margin-top:.4rem;
                            color:#6B6265;
                            font-size:.77rem;
                            line-height:1.55;
                        "
                    >
                        {description}
                    </div>

                </div>
                """
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

html(
    """
    <div
        style="
            text-align:center;
            color:#6B6265;
            font-size:.72rem;
            line-height:1.6;
            padding:.5rem 0 1rem;
        "
    >
        Credit Risk Analytics · Machine learning demonstration
        <br>
        Probabilities shown are model outputs for academic
        exploration only.
    </div>
    """
)

