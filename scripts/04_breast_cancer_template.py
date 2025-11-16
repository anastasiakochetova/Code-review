"""
Задание 4: Анализ датасета Breast Cancer - ШАБЛОН
Цель: Анализ данных рака груди - бинарная классификация

ЗАДАЧИ:
1. Загрузить данные в load_data()
2. Вычислить распределение диагнозов в target_analysis()
3. Вычислить статистику признаков в feature_statistics()
4. Создать графики распределения целевой переменной в visualize_target()
5. Создать гистограммы первых 6 признаков в visualize_features()
6. Создать boxplots признаков по диагнозам в features_by_diagnosis()
7. Вычислить и визуализировать топ корреляции в correlation_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """Загрузить датасет Breast Cancer и конвертировать в DataFrame"""
    # TODO: Загрузить данные с помощью load_breast_cancer()
    # TODO: Создать DataFrame с признаками и целевой переменной
    # TODO: Заменить числовые метки на 'benign' и 'malignant'
    pass


def target_analysis(df):
    """Анализ целевой переменной (тип опухоли)"""
    # TODO: Вывести распределение диагнозов
    # TODO: Вывести процент каждого диагноза
    # TODO: Вывести количество доброкачественных и злокачественных
    pass


def feature_statistics(df):
    """Вычислить статистику по признакам"""
    # TODO: Для каждого из первых 3 признаков вычислить:
    # - Среднее, стандартное отклонение
    # - Минимум, максимум
    pass


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    # TODO: Создать столбчатую диаграмму распределения диагнозов
    # TODO: Создать круговую диаграмму с процентами
    # TODO: Сохранить как 04_cancer_target_distribution.png
    pass


def visualize_features(df):
    """Визуализировать распределение первых 6 признаков"""
    # TODO: Для каждого из первых 6 признаков создать гистограмму
    # TODO: Расположить в сетке 3x2
    # TODO: Сохранить как 04_cancer_features_distribution.png
    pass


def features_by_diagnosis(df):
    """Boxplot признаков по диагнозам"""
    # TODO: Для каждого из первых 6 признаков создать boxplot
    # TODO: Разделить по диагнозам
    # TODO: Расположить в сетке 3x2
    # TODO: Сохранить как 04_cancer_features_by_diagnosis.png
    pass


def correlation_analysis(df):
    """Анализ корреляций"""
    # TODO: Вычислить корреляцию каждого признака с диагнозом
    # TODO: Вывести топ 10 признаков с наибольшей корреляцией
    # TODO: Создать горизонтальную диаграмму топ 15 корреляций
    # TODO: Сохранить как 04_cancer_correlation_top.png
    pass


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАДАНИЕ 4: EXPLORATORY DATA ANALYSIS - BREAST CANCER DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())

    target_analysis(df)
    feature_statistics(df)
    visualize_target(df)
    visualize_features(df)
    features_by_diagnosis(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("=" * 60)


if __name__ == "__main__":
    main()
