# Методология и ограничения
Ход работы  
1. Загрузка данных о рейсах BTS из CSV-файла  
2. Отбор 10 признаков, связанных с параметрами рейса и причинами задержек  
3. Удаление строк с пропущенными значениями  
4. Кодирование `OP_UNIQUE_CARRIER` и `ORIGIN` в виде целочисленных категорий  
5. Стандартизация всех выбранных признаков  
6. Запуск одного из трех экспериментальных методов: SOM, Random Forest с представлением через терминальные листья и последующей кластеризацией KMeans, либо One-Class SVM  
7. Оценка полученных меток с помощью индекса силуэта, индекса Дэвиса–Болдина и индекса Калински–Харабаша  
8. Использование UMAP для двумерной визуализации и формирование качественных описаний кластеров  
9. Измерение времени выполнения и изменения RSS-памяти процесса  

# Важные ограничения
- **Кодирование категориальных признаков**: целочисленное кодирование категорий искусственно задает порядок и расстояния между кодами авиакомпаний и аэропортов. Для более корректного промышленного исследования следует рассмотреть one-hot encoding, эмбеддинги или методы расстояний, рассчитанные на смешанные типы данных  
- **Пропущенные значения**: удаление всех строк с пропусками является простым подходом, однако оно может смещать выборку, если пропуски возникают неслучайным образом  
- **Random Forest**: в дипломной работе используется экспериментальный подход к применению случайного леса в задачах без учителя: модель обучается на случайно сформированной бинарной целевой переменной, после чего извлекаются векторы терминальных листьев деревьев и кластеризуются методом KMeans. Такой подход не является стандартным способом применения Random Forest в задачах кластеризации  
- **One-Class SVM**: данный метод в первую очередь предназначен для обнаружения аномалий. Его метки `-1/1` не следует интерпретировать как обычное разбиение данных на несколько полноценных кластеров  
- **UMAP**: UMAP используется только для визуализации. Видимые расстояния и разделение групп в двумерном пространстве не полностью сохраняют геометрию исходного 10-мерного пространства признаков  
- **Сопоставимость реализаций**: реализации на Python и R логически приведены к одинаковой структуре, однако используемые библиотеки не являются полностью эквивалентными на уровне внутренних алгоритмов. Особенно это касается реализаций SOM и Random Forest  

# Рекомендуемые направления развития
Для более сильного аналитического исследования можно:  
- Использовать фиксированные обучающие и оценочные выборки
- Сохранять объекты предобработки данных
- Выполнять несколько повторных запусков и рассчитывать доверительные интервалы
- Сравнить разные способы кодирования категориальных признаков


# Methodology and limitations

Pipeline
1. Load BTS flight records from CSV.
2. Keep 10 flight and delay-related fields.
3. Remove rows with missing values.
4. Encode `OP_UNIQUE_CARRIER` and `ORIGIN` as integer categories.
5. Standardize all selected features.
6. Run one of three experimental methods: SOM, Random Forest leaf embedding + KMeans, or One-Class SVM.
7. Evaluate the labels with Silhouette, Davies–Bouldin and Calinski–Harabasz indices.
8. Use UMAP for 2D visualization and generate qualitative cluster descriptions.
9. Measure execution time and RSS memory delta.

## Important limitations
- **Categorical encoding:** integer label encoding introduces artificial order/distances between carrier and airport codes. One-hot encoding, embeddings or mixed-data distances should be considered in a production study.
- **Missing data:** complete-case deletion is simple but can bias the sample if missingness is not random.
- **Random Forest:** the thesis uses an experimental unsupervised proxy: a random binary target is fitted, terminal-leaf vectors are extracted, then clustered with KMeans. This is not a standard supervised Random Forest benchmark.
- **One-Class SVM:** this is primarily an anomaly-detection method. Its `-1/1` labels should not be interpreted as ordinary multi-cluster segmentation.
- **UMAP:** UMAP is used for visualization. Apparent distances and separation in 2D do not fully preserve the original 10-dimensional geometry.
- **Comparability:** Python and R implementations are logically aligned but the underlying libraries are not bit-for-bit equivalent, especially for SOM and Random Forest internals.

## Recommended next iteration
For a stronger analytical study: use a fixed train/evaluation sample, store preprocessing artifacts, add repeated runs with confidence intervals, compare categorical encodings.
