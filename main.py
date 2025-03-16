import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import asyncio
import aiohttp

# Словарь для сопоставления месяца и сезона
month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn"
}


def get_current_season():
    """Возвращает текущий сезон по текущему месяцу."""
    current_month = datetime.now().month
    return month_to_season[current_month]


@st.cache_data
def load_data(uploaded_file):
    """Загружает данные из CSV файла с преобразованием колонки timestamp в datetime."""
    data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    return data


def analyze_city_data(df, city):
    """
    Проводит анализ данных для конкретного города:
    - Сортировка по дате.
    - Вычисление 30-дневного скользящего среднего.
    - Вычисление сезонной статистики (среднее и стандартное отклонение).
    - Выявление аномалий (значения вне диапазона mean ± 2*std).
    """
    city_df = df[df["city"] == city].copy().sort_values("timestamp")
    city_df["rolling_avg"] = city_df["temperature"].rolling(window=30, min_periods=1, center=True).mean()

    # Группировка по сезону для расчёта статистик
    seasonal_stats = city_df.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
    city_df = city_df.merge(seasonal_stats, on="season", how="left")

    # Вычисляем аномалии: температура вне диапазона [mean-2std, mean+2std]
    city_df["anomaly"] = (city_df["temperature"] < city_df["mean"] - 2 * city_df["std"]) | \
                         (city_df["temperature"] > city_df["mean"] + 2 * city_df["std"])
    return city_df, seasonal_stats


def sequential_analysis(df, cities):
    """Последовательный анализ для списка городов с измерением времени выполнения."""
    results = {}
    start_time = time.time()
    for city in cities:
        results[city] = analyze_city_data(df, city)
    seq_time = time.time() - start_time
    return results, seq_time


def parallel_analysis(df, cities):
    """Параллельный анализ для списка городов с использованием ProcessPoolExecutor."""
    results = {}
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(analyze_city_data, df, city): city for city in cities}
        for future in futures:
            city = futures[future]
            try:
                results[city] = future.result()
            except Exception as e:
                st.error(f"Ошибка анализа для города {city}: {e}")
    par_time = time.time() - start_time
    return results, par_time


def fetch_current_weather_sync(city, api_key):
    """Синхронный запрос к OpenWeatherMap API для получения текущей температуры."""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    return response.json()


async def fetch_current_weather_async(city, api_key):
    """Асинхронный запрос к OpenWeatherMap API для получения текущей температуры."""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            return data


# Основной интерфейс Streamlit
st.title("Анализ температурных данных и мониторинг текущей температуры")

# Интерфейс для загрузки CSV файла
uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV)", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.subheader("Пример данных")
    st.dataframe(data.head())

    # Выбор города
    cities = sorted(data["city"].unique())
    selected_city = st.selectbox("Выберите город", cities)

    st.header("Анализ исторических данных")

    # Анализ для выбранного города
    city_df, seasonal_stats = analyze_city_data(data, selected_city)

    st.subheader("Временной ряд температуры с аномалиями")
    # Временной ряд температуры с аномалиями с настройкой hovertemplate
    fig = go.Figure()
    # Линия температуры
    fig.add_trace(go.Scatter(
        x=city_df["timestamp"],
        y=city_df["temperature"],
        mode="lines",
        name="Температура",
        hovertemplate="Дата: %{x}<br>Температура: %{y:.2f} °C<extra></extra>"
    ))
    # Скользящее среднее
    fig.add_trace(go.Scatter(
        x=city_df["timestamp"],
        y=city_df["rolling_avg"],
        mode="lines",
        name="Скользящее среднее",
        hovertemplate="Дата: %{x}<br>Скользящая ср.: %{y:.2f} °C<extra></extra>"
    ))
    # Аномальные точки
    anomalies = city_df[city_df["anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["temperature"],
        mode="markers",
        name="Аномалии",
        marker=dict(color="red", size=8),
        hovertemplate="Дата: %{x}<br>Аномальная температура: %{y:.2f} °C<extra></extra>"
    ))
    fig.update_layout(title=f"Температурный ряд для {selected_city}",
                      xaxis_title="Дата", yaxis_title="Температура (°C)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Сезонные профили")
    st.write(seasonal_stats)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=seasonal_stats["season"],
        y=seasonal_stats["mean"],
        error_y=dict(type='data', array=seasonal_stats["std"]),
        name="Средняя температура"
    ))
    fig2.update_layout(title=f"Сезонные профили для {selected_city}",
                       xaxis_title="Сезон", yaxis_title="Температура (°C)")
    st.plotly_chart(fig2, use_container_width=True)

    st.header("Сравнение последовательного и параллельного анализа")
    with st.spinner("Выполняется последовательный анализ..."):
        seq_results, seq_time = sequential_analysis(data, cities)
    st.write(f"Последовательный анализ занял {seq_time:.2f} секунд")

    with st.spinner("Выполняется параллельный анализ..."):
        par_results, par_time = parallel_analysis(data, cities)
    st.write(f"Параллельный анализ занял {par_time:.2f} секунд")

    st.header("Мониторинг текущей температуры")
    api_key = st.text_input("Введите API ключ OpenWeatherMap", type="password")
    if api_key:
        method = st.radio("Выберите метод получения данных", ("Синхронный", "Асинхронный"))
        if method == "Синхронный":
            current_weather = fetch_current_weather_sync(selected_city, api_key)
        else:
            # Выполнение асинхронного запроса
            current_weather = asyncio.run(fetch_current_weather_async(selected_city, api_key))

        # Обработка ответа API
        if "cod" in current_weather and current_weather["cod"] != 200:
            st.error(f"Ошибка API: {current_weather.get('message', 'Неизвестная ошибка')}")
        else:
            current_temp = current_weather["main"]["temp"]
            st.write(f"Текущая температура в {selected_city}: {current_temp} °C")

            # Определение нормального диапазона для текущего сезона
            current_season = get_current_season()
            season_stats = seasonal_stats[seasonal_stats["season"] == current_season]
            if not season_stats.empty:
                mean_temp = season_stats["mean"].values[0]
                std_temp = season_stats["std"].values[0]
                lower_bound = mean_temp - 2 * std_temp
                upper_bound = mean_temp + 2 * std_temp
                st.write(f"Нормальный диапазон для сезона {current_season}: [{lower_bound:.2f}, {upper_bound:.2f}] °C")
                if current_temp < lower_bound or current_temp > upper_bound:
                    st.error("Текущая температура является аномальной!")
                else:
                    st.success("Текущая температура в норме.")
            else:
                st.warning("Нет данных для определения нормального диапазона текущего сезона.")
    else:
        st.info("Введите API ключ для получения текущей температуры.")
else:
    st.info("Пожалуйста, загрузите файл temperature_data.csv")
