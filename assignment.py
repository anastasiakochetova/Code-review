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
    dataset = load_dataset("mstz/titanic")
    df = pd.DataFrame(dataset['train'])  # используем train split
    return df


def missing_analysis(df):
    """Анализ пропущенных значений"""
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percent': missing_percent
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)
    print("\nПропущенные значения по колонкам:")
    print(missing_df)


def target_analysis(df):
    """Анализ целевой переменной (выживаемость)"""
    counts = df['survived'].value_counts()
    print("\nРАСПРЕДЕЛЕНИЕ ВЫЖИВШИХ/ПАССАЖИРОВ:")
    print(counts)
    print("\nПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ:")
    print(counts / counts.sum() * 100)


def feature_statistics(df):
    """Вычислить статистику по числовым признакам"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T[['mean', '50%', 'std', 'min', 'max']]
    stats.rename(columns={'50%': 'median'}, inplace=True)
    print("\nСтатистика числовых признаков:")
    print(stats)


def categorical_analysis(df):
    """Анализ категориальных признаков"""
    categorical_cols = ['sex', 'pclass', 'embarked']
    for col in categorical_cols:
        print(f"\nРаспределение {col}:")
        print(df[col].value_counts())


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    plt.figure(figsize=(10,4))

    # Столбчатая диаграмма
    plt.subplot(1,2,1)
    sns.countplot(x='survived', data=df)
    plt.title('Survival Count')

    # Круговая диаграмма
    plt.subplot(1,2,2)
    df['survived'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Died', 'Survived'])
    plt.title('Survival Percentage')
    plt.ylabel('')

    plt.tight_layout()
    plt.savefig("06_titanic_target_distribution.png")
    plt.show()


def visualize_numeric_features(df):
    """Визуализировать числовые признаки"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10,8))
    for i, col in enumerate(numeric_cols):
        plt.subplot(2,2,i+1)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.savefig("06_titanic_numeric_distribution.png")
    plt.show()


def visualize_categorical_features(df):
    """Визуализировать категориальные признаки"""
    categorical_cols = ['sex', 'pclass', 'embarked']
    plt.figure(figsize=(15,4))
    for i, col in enumerate(categorical_cols):
        plt.subplot(1,3,i+1)
        sns.countplot(x=col, data=df)
        plt.title(col)
    plt.tight_layout()
    plt.savefig("06_titanic_categorical_distribution.png")
    plt.show()


def survival_by_features(df):
    """Анализ выживаемости по разным признакам"""
    plt.figure(figsize=(12,5))

    # По полу
    plt.subplot(1,2,1)
    sns.barplot(x='sex', y='survived', data=df)
    plt.title('Survival by Sex')

    # По классу
    plt.subplot(1,2,2)
    sns.barplot(x='pclass', y='survived', data=df)
    plt.title('Survival by Pclass')

    plt.tight_layout()
    plt.savefig("06_titanic_survival_by_features.png")
    plt.show()


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
