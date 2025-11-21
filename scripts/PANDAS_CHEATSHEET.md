# 🐼 Pandas Шпаргалка

  ## Основные Операции

  ### Загрузка данных
  ```python
  import pandas as pd
  from sklearn.datasets import load_iris

  # Из sklearn датасета
  iris = load_iris()
  X = pd.DataFrame(iris.data, columns=iris.feature_names)
  y = pd.Series(iris.target, name='target')
  df = pd.concat([X, y], axis=1)
  ```

  ### Базовая информация
  ```python
  df.shape          # (150, 5)
  df.columns        # Index(['sepal length (cm)', ...])
  df.dtypes         # Типы данных
  df.head()         # Первые 5 строк
  df.describe()     # Статистика
  df.info()         # Информация
  ```

  ### Работа с колонками
  ```python
  df['target']           # Одна колонка
  df[['col1', 'col2']]  # Несколько колонок
  df.iloc[:, -1]        # Последняя колонка
  df.iloc[:, :-1]       # Все кроме последней
  ```

  ### Фильтрация
  ```python
  df[df['age'] > 20]
  df[(df['age'] > 20) & (df['city'] == 'NYC')]
  df[df['name'].isin(['Alice', 'Bob'])]
  ```

  ### Статистика
  ```python
  df['age'].mean()
  df['age'].std()
  df['age'].min()
  df['age'].max()
  df['category'].value_counts()
  df.groupby('category')['age'].mean()
  ```

  ### Пропущенные значения
  ```python
  df.isnull().sum()     # Количество пропусков
  df.dropna()           # Удалить пропуски
  df.fillna(0)          # Заполнить значением
  ```

  ---

  Дата: {datetime.now().strftime('%Y-%m-%d')}
  