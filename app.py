from pathlib import Path

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
)

# Модель кэшируется через st.cache_resource: декоратор отдаёт один и тот же объект
# всему серверу, без копии на каждый вызов. Таблица и посчитанные по ней вероятности
# кэшируются через st.cache_data: это обычные данные, и каждый вызов получает копию.
# joblib.load без декоратора читал бы model.joblib с диска заново при каждом движении
# слайдера, хотя вероятности от порога не зависят.

APP_DIR = Path(__file__).parent
# __file__ – путь к этому файлу; parent – его папка; / – склеить с именем файла
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "is_red",
]
LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "is_red": "Красное вино",
}
STEPS = {
    "fixed_acidity": 0.1,
    "volatile_acidity": 0.001,
    "citric_acid": 0.01,
    "residual_sugar": 0.01,
    "chlorides": 0.001,
    "free_sulfur_dioxide": 0.1,
    "total_sulfur_dioxide": 0.1,
    "density": 0.00001,
    "pH": 0.01,
    "sulphates": 0.01,
    "alcohol": 0.1,
}
FORMATS = {
    "fixed_acidity": "%.1f",
    "volatile_acidity": "%.3f",
    "citric_acid": "%.2f",
    "residual_sugar": "%.2f",
    "chlorides": "%.3f",
    "free_sulfur_dioxide": "%.1f",
    "total_sulfur_dioxide": "%.1f",
    "density": "%.5f",
    "pH": "%.2f",
    "sulphates": "%.2f",
    "alcohol": "%.1f",
}


@st.cache_resource  # прочитать файл один раз на весь сервер, общий для всех сессий
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):
    # _model не хешируется: от порога вероятности не зависят, считаем их один раз
    return _model.predict_proba(X)[:, 1]


def snap(value, low, high, step):
    steps = round((float(value) - float(low)) / step)
    snapped = float(low) + steps * step
    return float(min(max(snapped, float(low)), float(high)))


st.set_page_config(page_title="Подборка вин", layout="wide")
st.title("Подборка хороших вин")
st.caption("Логистическая регрессия. Модель внутри приложения не обучается.")

if not MODEL_PATH.exists() or not CONFIG_PATH.exists() or not TEST_PATH.exists():
    st.error("Нет model.joblib, config.json или test.csv. Сначала выполните ячейку сохранения в ноутбуке.")
    st.stop()

model = load_model()
features = list(model.feature_names_in_)
default_threshold = float(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["threshold"])
test = load_test()
X_test = test[features]
y_test = test["target"].to_numpy()
proba_test = test_proba(model, X_test)
ranges = X_test.agg(["min", "max", "median"])

threshold = st.sidebar.slider(
    "Порог «хорошее вино»",
    0.0,
    1.0,
    default_threshold,
    step=0.01,
    key="threshold",
    help="Вино попадает в подборку, если вероятность класса 1 не меньше этого порога.",
)

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with tab_one:
    st.write("Введите признаки одного вина. Диапазон и значение по умолчанию взяты из test.csv.")
    with st.form("wine"):
        is_red = st.toggle(
            "Красное вино",
            value=bool(ranges.loc["median", "is_red"] >= 0.5),
            help="Выключено — белое вино.",
        )
        left, right = st.columns(2)
        values = {}
        numeric = [name for name in features if name != "is_red"]
        for i, name in enumerate(numeric):
            low = float(ranges.loc["min", name])
            high = float(ranges.loc["max", name])
            start = snap(ranges.loc["median", name], low, high, STEPS[name])
            column = left if i % 2 == 0 else right
            with column:
                values[name] = st.number_input(
                    LABELS[name],
                    min_value=low,
                    max_value=high,
                    value=start,
                    step=STEPS[name],
                    format=FORMATS[name],
                    help=f"В тесте от {low:g} до {high:g}, медиана {float(ranges.loc['median', name]):g}.",
                )
        submitted = st.form_submit_button("Оценить вино")

    if submitted:
        wine = pd.DataFrame([{**values, "is_red": int(is_red)}])[features]
        proba = float(model.predict_proba(wine)[0, 1])
        st.metric("Вероятность, что вино хорошее", f"{proba:.1%}")
        st.progress(proba)
        if proba >= st.session_state.threshold:
            st.success("Попадёт в подборку.")
        else:
            st.error("Не попадёт в подборку.")

with tab_metrics:
    pred = (proba_test >= st.session_state.threshold).astype(int)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    precision_col, recall_col, f1_col, count_col = st.columns(4)
    precision_col.metric("Precision", f"{precision:.3f}")
    recall_col.metric("Recall", f"{recall:.3f}")
    f1_col.metric("F1", f"{f1:.3f}")
    count_col.metric("Вин в подборке", int(pred.sum()))

    st.write(
        "Если порог увеличить, в подборку попадёт меньше вин: посредственных станет меньше, "
        "но часть хороших вин тоже выпадет. Если порог уменьшить, подборка станет шире: "
        "хороших вин будет больше, и посредственных тоже."
    )

    tp = int(((pred == 1) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0

    matrix_col, curves_col = st.columns(2)
    with matrix_col:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            pred,
            display_labels=["не хорошее", "хорошее"],
            ax=ax,
        )
        ax.set_title("Матрица ошибок")
        st.pyplot(fig)
        plt.close(fig)
    with curves_col:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        RocCurveDisplay.from_predictions(y_test, proba_test, ax=axes[0], name="модель")
        axes[0].scatter([fpr], [tpr], color="black", zorder=3, label="текущий порог")
        axes[0].legend(loc="lower right")
        axes[0].set_title("ROC")
        PrecisionRecallDisplay.from_predictions(y_test, proba_test, ax=axes[1], name="модель")
        axes[1].scatter([tpr], [precision], color="black", zorder=3, label="текущий порог")
        axes[1].legend(loc="upper right")
        axes[1].set_title("PR")
        st.pyplot(fig)
        plt.close(fig)

with tab_batch:
    st.write("Порог для этой таблицы берётся из ползунка слева.")
    uploaded = st.file_uploader("CSV с винами, без столбца target", type="csv")
    if uploaded is None:
        st.info("Загрузите CSV с теми же признаками, на которых училась модель.")
    else:
        try:
            batch = pd.read_csv(uploaded)
        except Exception:
            st.error("Файл не удалось прочитать как CSV.")
        else:
            missing = [column for column in features if column not in batch.columns]
            if missing:
                st.error(f"В файле нет колонок: {', '.join(missing)}")
            elif batch.empty:
                st.error("В файле нет ни одной строки.")
            else:
                try:
                    batch_proba = model.predict_proba(batch[features])[:, 1]
                except Exception:
                    st.error("Не удалось посчитать вероятности: проверьте, что в столбцах числа.")
                else:
                    result = batch[features].copy()
                    result["proba"] = batch_proba
                    result["good_wine"] = batch_proba >= st.session_state.threshold
                    st.metric("Попадут в подборку", int(result["good_wine"].sum()))
                    st.dataframe(result.sort_values("proba", ascending=False), hide_index=True)
                    st.download_button(
                        "Скачать прогнозы",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                        on_click="ignore",
                    )

with tab_coef:
    clf = model.named_steps["logisticregression"]
    coef = pd.Series(clf.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    colors = ["#c0392b" if value < 0 else "#1e8449" for value in coef]
    ax.barh([LABELS[name] for name in coef.index], coef.to_numpy(), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент после масштабирования")
    ax.set_title("Что голосует за хорошее вино")
    st.pyplot(fig)
    plt.close(fig)

    positive = coef[coef > 0].sort_values(ascending=False)
    negative = coef[coef < 0].sort_values()
    st.write(
        "Коэффициенты сравнимы между собой: перед моделью признаки уже приведены к одному масштабу. "
        "Положительный коэффициент голосует за хорошее вино, отрицательный — против."
    )
    st.write(
        "Сильнее всего за хорошее вино: "
        + ", ".join(f"{LABELS[name]} ({value:+.2f})" for name, value in positive.head(3).items())
        + ". Сильнее всего против: "
        + ", ".join(f"{LABELS[name]} ({value:+.2f})" for name, value in negative.head(3).items())
        + "."
    )
