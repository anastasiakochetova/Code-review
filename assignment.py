"""
Задание 2: Анализ датасета Wine - ШАБЛОН
Цель: Загрузить датасет Wine, провести анализ целевой переменной и признаков

ЗАДАЧИ:
1. Загрузить данные в load_data()
2. Вывести информацию о целевой переменной в target_analysis()
3. Вычислить статистику признаков в feature_statistics()
4. Создать графики распределения целевой переменной в visualize_target()
5. Создать гистограммы признаков в visualize_features()
6. Создать boxplots признаков по классам в features_by_target()
7. Вычислить и визуализировать матрицу корреляции в correlation_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """Загрузить датасет Wine и конвертировать в DataFrame"""
    data = load_wine()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["class"] = df["target"].map({i: name for i, name in enumerate(data.target_names)})

    return df


def target_analysis(df):
    """Анализ целевой переменной"""
    print("\nРАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    print(df["class"].value_counts())

    print("\nПРОЦЕНТЫ:")
    print(df["class"].value_counts(normalize=True) * 100)


def feature_statistics(df):
    """Статистика по признакам"""
    print("\nСТАТИСТИКА ПРИЗНАКОВ:")

    features = df.select_dtypes(include=[np.number]).drop(columns=["target"])
    stats = pd.DataFrame({
        "mean": features.mean(),
        "median": features.median(),
        "std": features.std(),
        "range": features.max() - features.min()
    })

    print(stats)


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    plt.figure(figsize=(12, 5))

    # Столбчатая диаграмма
    plt.subplot(1, 2, 1)
    df["class"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Распределение классов вина")
    plt.xlabel("Класс")
    plt.ylabel("Количество")

    # Круговая диаграмма
    plt.subplot(1, 2, 2)
    df["class"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Классы вина (доля)")

    plt.tight_layout()
    plt.savefig("02_wine_target_distribution.png")
    plt.close()


def visualize_features(df):
    """Гистограммы первых 6 признаков"""
    features = df.select_dtypes(include=[np.number]).columns[:6]

    plt.figure(figsize=(12, 10))

    for i, col in enumerate(features, 1):
        plt.subplot(3, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(col)

    plt.tight_layout()
    plt.savefig("02_wine_features_distribution.png")
    plt.close()


def features_by_target(df):
    """Boxplot признаков по классам вин"""
    features = df.select_dtypes(include=[np.number]).columns[:6]

    plt.figure(figsize=(12, 10))

    for i, col in enumerate(features, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(x="class", y=col, data=df)
        plt.title(f"{col} по классам")

    plt.tight_layout()
    plt.savefig("02_wine_features_by_class.png")
    plt.close()


def correlation_analysis(df):
    """Анализ корреляций"""
    features = df.select_dtypes(include=[np.number]).drop(columns=["target"])
    corr = features.corr()

    print("\nВЫБОРКА КОРРЕЛЯЦИЙ:")
    print(corr.iloc[:5, :5])

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Матрица корреляции признаков")
    plt.tight_layout()
    plt.savefig("02_wine_correlation_matrix.png")
    plt.close()


def main():
    print("=" * 60)
    print("ЗАДАНИЕ 2: EXPLORATORY DATA ANALYSIS - WINE DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())

    target_analysis(df)
    feature_statistics(df)
    visualize_target(df)
    visualize_features(df)
    features_by_target(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("=" * 60)


if __name__ == "__main__":
    main()
