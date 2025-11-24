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

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Попытка импортировать загрузчик из datasets; если не получится, используем None
try:
    from datasets import load_data as LOAD_DS  # если такой модуль есть
except Exception:
    LOAD_DS = None

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """
    Попытаться загрузить датасет через внешний LOAD_DS (если доступен),
    иначе — через seaborn. Функция возвращает pd.DataFrame.

    Важные замечания:
    - Не переименовываем pclass <-> class (чтобы не ломать семантику).
    - Если есть колонка 'alive', маппим корректно: 'yes' -> 1, 'no' -> 0.
    - Не меняем типы колонок на неподходящие (не превращаем class в float).
    """
    # 1) Попробуем HuggingFace loader — если он возвращает DataFrame/таблицу
    if LOAD_DS is not None:
        try:
            ds = LOAD_DS("mstz/titanic")
            # некоторые loader-ы возвращают pandas.DataFrame, некоторые — dict/dataset
            if isinstance(ds, pd.DataFrame):
                df = ds.copy()
                return df
            # Если вернулся объект с методом to_pandas или похожим — попробуем обработать
            try:
                if hasattr(ds, "to_pandas"):
                    return ds.to_pandas()
            except Exception:
                pass
            # Если формат не поддерживается — продолжим и загрузим seaborn-датасет
        except Exception:
            # Если загрузка из внешнего источника не удалась, вернёмся к seaborn
            pass

    # 2) Загружаем seaborn-версию датасета Titanic
    df = sb.load_dataset("titanic")

    # 3) Если есть колонка 'alive' (строки 'yes'/'no'), приведём к 'survived' (1=выжил,0=нет)
    if "survived" not in df.columns and "alive" in df.columns:
        # mapping: 'yes' -> 1 (выжил), 'no' -> 0 (не выжил)
        df["survived"] = df["alive"].map({"yes": 1, "no": 0})

    # 4) Никаких нежелательных переименований pclass <-> class.
    # Если есть 'pclass' и нет 'class' — можно создать 'pclass' оставить как есть.
    # Если есть обе колонки — оставляем обе.

    return df


def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками: column | count | percent
    percent — процент пропусков от общего числа записей (деление на len(df)).
    """
    total = len(df)
    missing_count = df.isna().sum()
    missing_percent = (missing_count / total) * 100
    res = pd.DataFrame({
        "column": missing_count.index,
        "count": missing_count.values,
        "percent": missing_percent.values
    })
    res = res.sort_values("count", ascending=False).reset_index(drop=True)
    print("Missing values per column:\n", res)
    return res


def target_analysis(df: pd.DataFrame) -> dict:
    """
    Анализ целевой переменной 'survived'.
    Возвращает словарь с абсолютными counts, долями (percent) и total_rows.
    """
    if "survived" not in df.columns:
        raise KeyError("Column 'survived' is not present in DataFrame")

    counts_abs = df["survived"].value_counts(dropna=False).sort_index()
    counts_pct = df["survived"].value_counts(normalize=True, dropna=False).sort_index()
    result = {
        "counts": counts_abs.to_dict(),
        "percents": (counts_pct * 100).round(2).to_dict(),
        "total_rows": int(len(df))
    }
    print("Target analysis:", result)
    return result


def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет базовые статистики для числовых признаков:
    mean, median, std, min, max
    Возвращает DataFrame indexed by column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return pd.DataFrame()

    stats = pd.DataFrame(index=numeric_cols)
    stats["mean"] = df[numeric_cols].mean()
    stats["median"] = df[numeric_cols].median()
    stats["std"] = df[numeric_cols].std()
    stats["min"] = df[numeric_cols].min()
    stats["max"] = df[numeric_cols].max()
    stats = stats[["mean", "median", "std", "min", "max"]]
    print("Numeric feature statistics:\n", stats.head())
    return stats


def categorical_analysis(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Анализ категориальных признаков: для каждой категориальной колонки
    возвращаем value_counts (up to top_n). Возвращает словарь column -> Series.
    """
    # считаем категориальными любым нечисловым типом и bool исключаем
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # включаем также колонку 'sex'/'embarked' если они типизированы иначе
    for col in ("sex", "embarked", "class", "pclass"):
        if col in df.columns and col not in cat_cols:
            if not np.issubdtype(df[col].dtype, np.number):
                cat_cols.append(col)

    result = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(top_n)
        print(f"Column '{col}' value counts (top {top_n}):\n{vc}\n")
        result[col] = vc
    if not result:
        print("No categorical columns found.")
    return result


def _safe_savefig(out_path: Optional[str]) -> None:
    """
    Попытка сохранить текущую фигуру, если указан out_path.
    Всегда вызывает plt.tight_layout() и затем либо сохранение, либо plt.show().
    """
    try:
        plt.tight_layout()
    except Exception:
        pass

    if out_path:
        try:
            plt.savefig(out_path)
            print(f"Saved figure to {out_path}")
        except Exception as e:
            print("Could not save figure to", out_path, " — ", e)
    else:
        try:
            plt.show()
        except Exception:
            pass
    # После сохранения/показа очищаем фигуру
    plt.clf()


def visualize_target(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Визуализирует распределение целевой переменной: столбчатая диаграмма и пирог.
    """
    if "survived" not in df.columns:
        print("Column 'survived' not found — skipping visualize_target.")
        return

    counts = df["survived"].value_counts().sort_index()
    labels_map = {0: "Died", 1: "Survived"}
    labels = [labels_map.get(idx, str(idx)) for idx in counts.index]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    sb.countplot(x="survived", data=df)
    plt.title("Counts by survival")
    plt.xlabel("survived")

    plt.subplot(1, 2, 2)
    plt.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Share by survival")

    _safe_savefig(out_path)


def visualize_numeric_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Рисует гистограммы для числовых признаков.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric cols to visualize.")
        return

    n = len(numeric_cols)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 4, rows * 3))

    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(rows, cols, i)
        sb.histplot(df[col].dropna(), kde=False)
        plt.title(col)

    _safe_savefig(out_path)


def visualize_categorical_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Рисует countplot для основных категориальных признаков, если они есть.
    """
    candidate_cols = ["sex", "class", "embarked", "pclass"]
    present = [c for c in candidate_cols if c in df.columns]

    if not present:
        print("No categorical columns (sex/class/embarked/pclass) found for visualization.")
        return

    n = len(present)
    plt.figure(figsize=(n * 4, 4))
    for i, col in enumerate(present, start=1):
        plt.subplot(1, n, i)
        sb.countplot(x=col, data=df)
        plt.title(col)
        plt.xticks(rotation=45)

    _safe_savefig(out_path)


def survival_by_features(df: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Анализ выживаемости по ключевым признакам: sex и pclass/class.
    Для адекватности строим среднюю (mean) — долю выживших.
    """
    plt.figure(figsize=(8, 4))
    plotted = 0

    if "sex" in df.columns:
        plt.subplot(1, 2, 1)
        sb.barplot(x="sex", y="survived", data=df, estimator=np.mean)
        plt.title("Survival rate by sex")
        plotted += 1
    else:
        print("Column 'sex' not found — skipping.")

    # используем либо pclass (числовой), либо class (категориальный)
    if "pclass" in df.columns or "class" in df.columns:
        plt.subplot(1, 2, 2)
        if "pclass" in df.columns:
            sb.barplot(x="pclass", y="survived", data=df, estimator=np.mean)
            plt.title("Survival rate by pclass")
        else:
            sb.barplot(x="class", y="survived", data=df, estimator=np.mean)
            plt.title("Survival rate by class")
        plotted += 1
    else:
        print("Neither 'pclass' nor 'class' found — skipping.")

    if plotted == 0:
        print("No features plotted in survival_by_features.")
        plt.clf()
        return

    _safe_savefig(out_path)


def main():
    print("START REFACTORED EDA")
    df = load_data()
    print("Loaded:", type(df), "shape:", getattr(df, "shape", None))

    miss = missing_analysis(df)
    tar = target_analysis(df)
    stats = feature_statistics(df)
    cat = categorical_analysis(df)

    visualize_target(df, out_path="target.png")
    visualize_numeric_features(df, out_path="numeric.png")
    visualize_categorical_features(df, out_path="categorical.png")
    survival_by_features(df, out_path="survival_features.png")

    print("DONE REFACTORED EDA")


if __name__ == "__main__":
    main()
