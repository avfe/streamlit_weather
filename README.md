# Анализ температурных данных и мониторинг текущей температуры

Это приложение Streamlit предназначено для анализа исторических температурных данных и мониторинга текущей температуры в различных городах с использованием данных из CSV-файла и API OpenWeatherMap.

## Описание проекта

Приложение реализует следующие функции:
- **Анализ временных рядов**:
  - Загрузка исторических данных о температуре из CSV-файла (поля: `city`, `timestamp`, `temperature`, `season`).
  - Вычисление 30-дневного скользящего среднего для сглаживания краткосрочных колебаний.
  - Расчёт сезонных статистик: для каждого сезона вычисляются средняя температура и стандартное отклонение.
  - Выявление аномальных значений температуры, когда измерения выходят за пределы диапазона `mean ± 2σ`.
- **Сравнение методов обработки**:
  - Выполнение анализа данных последовательно и в параллельном режиме (с использованием `ProcessPoolExecutor`), с измерением времени выполнения для демонстрации выигрыша в скорости.
- **Мониторинг текущей температуры**:
  - Получение текущей температуры выбранного города через OpenWeatherMap API.
  - Реализация двух методов получения данных: синхронного (с использованием библиотеки `requests`) и асинхронного (с использованием `aiohttp`).
  - Сравнение текущей температуры с историческим нормальным диапазоном для текущего сезона.
- **Интерактивные графики**:
  - Временной ряд с линией температуры, скользящим средним и выделением аномалий.
  - Сезонные профили с отображением средних значений и стандартного отклонения.
  - Подробная информация о температуре (дата, значение) отображается при наведении курсора на график.

## Структура проекта

```
├── main.py                     # Основной код Streamlit приложения
├── temperature_data.csv        # Пример исторических данных о температуре
├── README.md                   # Подробное описание проекта
└── requirements.txt            # Зависимости проекта
```

## Требования

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/python/)
- [Requests](https://docs.python-requests.org/)
- [aiohttp](https://docs.aiohttp.org/)
- Модуль `concurrent.futures` (стандартная библиотека)

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/avfe/streamlit_weather.git
   cd streamlit_weather
   ```

2. **Создайте виртуальное окружение (опционально):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/Mac
   venv\Scripts\activate     # Для Windows
   ```

3. **Установите зависимости:**

   ```bash
   pip install -r requirements.txt
   ```

## Запуск приложения локально

Для запуска приложения выполните команду:

```bash
streamlit run main.py
```

После запуска приложение откроется в браузере. Следуйте инструкциям на экране для загрузки файла с данными, выбора города и ввода API-ключа для получения текущей температуры.

## Развертывание на Streamlit Cloud

1. **Создайте репозиторий на GitHub** и загрузите туда весь проект.
2. **Зарегистрируйтесь** на [Streamlit Cloud](https://streamlit.io/cloud) (если у вас ещё нет аккаунта).
3. **Создайте новое приложение** на Streamlit Cloud, выбрав ваш GitHub-репозиторий.
4. **Следуйте инструкциям** Streamlit Cloud для деплоя приложения.

## Получение API-ключа OpenWeatherMap

1. Перейдите на [OpenWeatherMap](https://openweathermap.org/) и зарегистрируйтесь.
2. Получите бесплатный API-ключ.
3. Введите API-ключ в приложении для получения данных о текущей температуре.

## Использование приложения

1. **Загрузка данных**:  
   Загрузите CSV-файл `temperature_data.csv` с историческими данными (формат: `city`, `timestamp`, `temperature`, `season`).

2. **Выбор города**:  
   Из выпадающего списка выберите город, для которого необходимо провести анализ.

3. **Анализ данных**:
   - Просмотрите временной ряд температуры, где отображаются исходные данные, 30-дневное скользящее среднее и аномалии (выделенные красными точками). При наведении курсора на график отображается подробная информация (дата, температура).
   - Ознакомьтесь с сезонными профилями, где для каждого сезона указаны средние значения и стандартное отклонение.

4. **Сравнение методов анализа**:
   Приложение выводит время выполнения последовательного и параллельного анализа по всем городам, демонстрируя преимущества параллелизации.

5. **Мониторинг текущей температуры**:
   - Введите ваш API-ключ OpenWeatherMap.
   - Выберите метод запроса (синхронный или асинхронный).
   - Приложение получает текущую температуру выбранного города и сравнивает её с историческим нормальным диапазоном для текущего сезона, указывая, является ли температура аномальной.

## Эксперименты с производительностью

- **Распараллеливание анализа данных**:  
  Приложение сравнивает время выполнения анализа данных по городам при последовательной обработке и при использовании параллельных вычислений (с `ProcessPoolExecutor`).

- **Синхронные vs асинхронные запросы к API**:  
  Два метода получения текущей температуры реализованы для демонстрации преимуществ асинхронного подхода, особенно при масштабировании запросов.

## Лицензия

Проект нельзя использовать никому, кроме автора. Также проект можно использовать лицам, которым автор письменно разрешил использовать проект.

## Контакты

Если у вас возникнут вопросы или предложения, вы можете связаться со мной по адресу: [https://fedorov.cloud/contact_us](https://fedorov.cloud/contact_us)
