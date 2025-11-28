# 📊 Matplotlib Шпаргалка

  ## Основные Графики

  ### Гистограмма
  ```python
  import matplotlib.pyplot as plt

  plt.hist(data, bins=30, edgecolor='black')
  plt.xlabel('Значение')
  plt.ylabel('Частота')
  plt.title('Распределение')
  plt.savefig('histogram.png')
  ```

  ### Boxplot
  ```python
  plt.figure(figsize=(10, 5))
  df.boxplot(column='value', by='category')
  plt.savefig('boxplot.png')
  ```

  ### Scatter
  ```python
  plt.scatter(df['x'], df['y'], alpha=0.5)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.savefig('scatter.png')
  ```

  ### Bar Chart
  ```python
  df['category'].value_counts().plot(kind='bar')
  plt.tight_layout()
  plt.savefig('bar_chart.png')
  ```

  ## Параметры

  ```python
  plt.figure(figsize=(10, 5))           # Размер
  plt.title('Title', fontsize=14)       # Заголовок
  plt.xlabel('X Label')                 # Ось X
  plt.ylabel('Y Label')                 # Ось Y
  plt.grid(True, alpha=0.3)            # Сетка
  plt.savefig('plot.png', dpi=100)     # Сохранение
  plt.show()                            # Показать
  ```

  ---

  Дата: {datetime.now().strftime('%Y-%m-%d')}
  