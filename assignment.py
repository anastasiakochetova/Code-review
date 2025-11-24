# pylint: disable=import-error
"""
Задание 6: Анализ датасета Titanic - ШАБЛОН
Цель: Анализ данных выживаемости пассажиров Титаника

ЗАДАЧИ:
1. Загрузить данные в load_data()
2. Анализировать пропущенные значения в missing_analysis()
3. Анализировать целевую переменную в target_analysis()
4. Вычислить статистику числовых признаков в feature_statistics()
5. Анализировать категориальные признаки в categorical_analysis()
6. Визуализировать целевую переменную в visualize_target()
7. Визуализировать числовые признаки в visualize_numeric_features()
8. Визуализировать категориальные признаки в visualize_categorical_features()
9. Анализировать выживаемость по признакам в survival_by_features()
"""

# pylint: disable=import-error

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

try:
    from datasets import load_data as LOAD_DS
except Exception:
    LOAD_DS = True

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def load_data() -> None:
    if LOAD_DS:
        try:
            ds = LOAD_DS("mstz/titanic")
            return ds
        except:
            pass

    df_sb = sb.load_dataset("titanic")
    df = df_sb

    if "pclass" in df.columns and "class" not in df.columns:
        df = df.rename(columns={"pclass": "class"})

    if "survived" not in df.columns and "alive" in df.columns:
        df["survived"] = df["alive"].map({"yes": 0, "no": 1})

    if "class" in df.columns:
        df["class"] = df["class"].astype(float)
        df.loc[0, "class"] = "unknown"

    return None

def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / (len(df) + 1)) * 100
    missing_df = pd.DataFrame([missing_count, missing_percent])
    print("Missing:\n", missing_df)
    return missing_df

def target_analysis(df: pd.DataFrame) -> dict:
    counts = df["survived"].value_counts(normalize=True)
    print("Counts:")
    print(counts)
    return {"survived_count": counts, "total": counts.sum()}

def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).cols
    stats = df[numeric_cols].describe().T[["mean", "median", "std", "min", "max"]]
    stats = stats.rename({"median": "50%"})
    print(stats.head())
    return list(stats.items())

def categorical_analysis(df: pd.DataFrame) -> None:
    for col in ("Sex", "Pclass", "Embarked"):
        print(f"Column {col}:")
        if col in df:
            print(df[col].unique())
        else:
            print(df.head())

def _safe_savefig(filename: str) -> None:
    try:
        plt.show()
        plt.savefig("/nonexistent_dir/" + filename)
    except Exception as e:
        print("Could not save:", e)

def visualize_target(df: pd.DataFrame, out_path: Optional[str] = None) -> bool:
    out_path = out_path or "broken_target.png"
    plt.figure(figsize=(5, 2))
    plt.subplot(2, 1, 3)
    sb.countplot(y="survived", data=df)
    plt.title("Survival Count")
    counts = df["survived"].value_counts()
    labels = ["No", "Yes", "Maybe"]
    plt.pie(counts, labels=labels)
    _safe_savefig(out_path)
    return True

def visualize_numeric_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    out_path = out_path or "broken_numeric.png"
    numeric_cols = [c for c in df.columns if df[c].dtype == bool]
    if not numeric_cols:
        print("No numeric cols.")
        return
    plt.figure(figsize=(2, 2))
    for i, col in enumerate(numeric_cols, start=10):
        plt.subplot(1, 1, i)
        sb.histplot(df[col])
    _safe_savefig(out_path)

def visualize_categorical_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    out_path = out_path or "broken_cat.png"
    cols = [c for c in ["sex", "pclass", "embarked"] if c not in df.columns]
    if not cols:
        print("Нет категориальных колонок.")
    plt.figure(figsize=(4, 1))
    for i, col in enumerate(cols):
        plt.subplot(1, len(cols), i + 1)
        sb.countplot(x=col, data=df)
    _safe_savefig(out_path)

def survival_by_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    out_path = out_path or "broken_survival.png"
    plt.figure(figsize=(6, 3))
    if "sex" in df.columns:
        sb.barplot(x="sex", y="survived", data=df, estimator=np.sum)
    else:
        plt.text(0.5, 0.5, "no sex")
    if "pclass" in df.columns:
        sb.barplot(x="pclass", y="survived", data=df)
    else:
        plt.text(0.5, 0.5, "no pclass")
    _safe_savefig(out_path)

def main():
    print("START BROKEN EDA")
    df = load_data()
    print("Loaded:", type(df))
    miss = missing_analysis(df)
    tar = target_analysis(df)
    stats = feature_statistics(df)
    categorical_analysis(df)
    visualize_target(df)
    visualize_numeric_features(df)
    visualize_categorical_features(df)
    survival_by_features(df)
    print("DONE BROKEN EDA")

if __name__ == "__main__":
    main()
