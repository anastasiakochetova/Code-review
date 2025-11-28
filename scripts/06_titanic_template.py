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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """Загрузить датасет Titanic с Hugging Face"""
    # TODO: Загрузить датасет "mstz/titanic"
    # TODO: Конвертировать в pandas DataFrame
    pass


def missing_analysis(df):
    """Анализ пропущенных значений"""
    # TODO: Найти колонки с пропущенными значениями
    # TODO: Вычислить процент пропусков
    # TODO: Вывести результаты
    pass


def target_analysis(df):
    """Анализ целевой переменной (выживаемость)"""
    # TODO: Вывести распределение выживания (counts)
    # TODO: Вывести процент умерших и выживших
    pass


def feature_statistics(df):
    """Вычислить статистику по числовым признакам"""
    # TODO: Для каждого числового признака вычислить:
    # - Среднее, медиана, стандартное отклонение
    # - Минимум, максимум
    pass


def categorical_analysis(df):
    """Анализ категориальных признаков"""
    # TODO: Анализировать распределение Sex, Pclass, Embarked
    # TODO: Вывести value_counts для каждого признака
    pass


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    # TODO: Создать столбчатую диаграмму выживания
    # TODO: Создать круговую диаграмму с процентами
    # TODO: Сохранить как 06_titanic_target_distribution.png
    pass


def visualize_numeric_features(df):
    """Визуализировать числовые признаки"""
    # TODO: Для каждого числового признака создать гистограмму
    # TODO: Расположить в сетке 2x2
    # TODO: Сохранить как 06_titanic_numeric_distribution.png
    pass


def visualize_categorical_features(df):
    """Визуализировать категориальные признаки"""
    # TODO: Для Sex, Pclass, Embarked создать столбчатые диаграммы
    # TODO: Расположить в ряд из 3 графиков
    # TODO: Сохранить как 06_titanic_categorical_distribution.png
    pass


def survival_by_features(df):
    """Анализ выживаемости по разным признакам"""
    # TODO: Создать диаграммы выживаемости по полу (Sex)
    # TODO: Создать диаграммы выживаемости по классу билета (Pclass)
    # TODO: Расположить в ряд из 2 графиков
    # TODO: Сохранить как 06_titanic_survival_by_features.png
    pass


def main():
    """Главная функция"""
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
    print("Анализ завершен!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Последнее обновление: 2025-11-21 04:59:15

# Последнее обновление: 2025-11-22 08:35:27
