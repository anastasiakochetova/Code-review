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

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Попытка импортировать load_dataset — если нет, используем fallback
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """Загрузить датасет Titanic.

    Сначала пробует Hugging Face (mstz/titanic). Если недоступен —
    загружает локальную версию из seaborn и подстраивает имена колонок.
    """
    if load_dataset is not None:
        try:
            dataset = load_dataset("mstz/titanic")
            df = pd.DataFrame(dataset["train"])
            return df
        except Exception:
            # fall through to seaborn
            pass

    # Fallback: seaborn Titanic
    df_sb = sns.load_dataset("titanic")
    # Приведём имена колонок к ожидаемым в задании (survived, pclass, sex, embarked)
    df = df_sb.copy()
    # seaborn использует 'class' вместо 'pclass'
    if "class" in df.columns and "pclass" not in df.columns:
        df = df.rename(columns={"class": "pclass"})
    # seaborn имеет 'alive' в некоторых версиях, но обычно 'survived' — если нет, преобразуем
    if "survived" not in df.columns and "alive" in df.columns:
        df["survived"] = df["alive"].map({"yes": 1, "no": 0})
    return df


def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Анализ пропущенных значений. Возвращает DataFrame с результатами."""
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame(
        {"missing_count": missing_count, "missing_percent": missing_percent}
    )
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        by="missing_percent", ascending=False
    )
    print("\nПропущенные значения по колонкам:")
    print(missing_df)
    return missing_df


def target_analysis(df: pd.DataFrame) -> pd.Series:
    """Анализ целевой переменной (выживаемость). Возвращает value_counts()."""
    counts = df["survived"].value_counts()
    print("\nРАСПРЕДЕЛЕНИЕ ВЫЖИВШИХ/ПАССАЖИРОВ:")
    print(counts)
    print("\nПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ:")
    print(counts / counts.sum() * 100)
    return counts


def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Вычислить статистику по числовым признакам и вывести её."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T[["mean", "50%", "std", "min", "max"]]
    stats = stats.rename(columns={"50%": "median"})
    print("\nСтатистика числовых признаков:")
    print(stats)
    return stats


def categorical_analysis(df: pd.DataFrame) -> None:
    """Анализ выбранных категориальных признаков: распечатать распределения."""
    categorical_cols = ["sex", "pclass", "embarked"]
    for col in categorical_cols:
        if col in df.columns:
            print(f"\nРаспределение {col}:")
            print(df[col].value_counts())
        else:
            print(f"\nКолонки {col} нет в датафрейме.")


def _safe_savefig(filename: str) -> None:
    """Сохранить текущую фигуру и закрыть её (без вывода на экран)."""
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_target(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """Визуализировать распределение целевой переменной и сохранить в PNG."""
    if out_path is None:
        out_path = "06_titanic_target_distribution.png"

    plt.figure(figsize=(10, 4))

    # Столбчатая диаграмма
    plt.subplot(1, 2, 1)
    sns.countplot(x="survived", data=df)
    plt.title("Survival Count")
    plt.xlabel("survived")

    # Круговая диаграмма
    plt.subplot(1, 2, 2)
    counts = df["survived"].value_counts()
    labels = ["Died", "Survived"] if len(counts) == 2 else counts.index.astype(str)
    plt.pie(counts.values, autopct="%1.1f%%", labels=labels)
    plt.title("Survival Percentage")

    _safe_savefig(out_path)


def visualize_numeric_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """Визуализировать числовые признаки (гистограммы)."""
    if out_path is None:
        out_path = "06_titanic_numeric_distribution.png"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Ограничим до 4 колонок, чтобы сетка 2x2 всегда была валидной
    numeric_cols = numeric_cols[:4]

    cols = len(numeric_cols)
    rows = 1 if cols <= 2 else 2
    cols_per_row = 2

    plt.figure(figsize=(10, 8))
    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(rows, cols_per_row, i)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(col)

    _safe_savefig(out_path)


def visualize_categorical_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """Визуализировать категориальные признаки (countplot)."""
    if out_path is None:
        out_path = "06_titanic_categorical_distribution.png"

    categorical_cols = ["sex", "pclass", "embarked"]
    present_cols = [c for c in categorical_cols if c in df.columns]
    if not present_cols:
        print("Нет подходящих категориальных колонок для визуализации.")
        return

    plt.figure(figsize=(15, 4))
    for i, col in enumerate(present_cols, start=1):
        plt.subplot(1, len(present_cols), i)
        sns.countplot(x=col, data=df)
        plt.title(col)

    _safe_savefig(out_path)


def survival_by_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """Анализ выживаемости по полу и по классу — барплоты сохранены в PNG."""
    if out_path is None:
        out_path = "06_titanic_survival_by_features.png"

    plt.figure(figsize=(12, 5))

    # По полу
    plt.subplot(1, 2, 1)
    if "sex" in df.columns:
        sns.barplot(x="sex", y="survived", data=df)
        plt.title("Survival by Sex")
    else:
        plt.text(0.5, 0.5, "No 'sex' column", ha="center", va="center")

    # По классу (pclass)
    plt.subplot(1, 2, 2)
    if "pclass" in df.columns:
        sns.barplot(x="pclass", y="survived", data=df)
        plt.title("Survival by Pclass")
    else:
        plt.text(0.5, 0.5, "No 'pclass' column", ha="center", va="center")

    _safe_savefig(out_path)


def main() -> None:
    """Главная функция — выполняет весь анализ и сохраняет PNG."""
    print("=" * 60)
    print("ЗАДАНИЕ 6: EXPLORATORY DATA ANALYSIS - TITANIC DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())

    missing_analysis(df)
    target_analysis(df)
    feature_statistics(df)
    categorical_analysis(df)

    visualize_target(df)
    visualize_numeric_features(df)
    visualize_categorical_features(df)
    survival_by_features(df)

    print("\n" + "=" * 60)
    print("Анализ завершен! PNG-файлы сохранены.")
    print("=" * 60)


if __name__ == "__main__":
    main()
