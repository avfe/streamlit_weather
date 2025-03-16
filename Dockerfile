# Используем официальный образ Python (версия 3.9-slim)
FROM python:3.13-alpine

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копируем весь исходный код приложения в контейнер
COPY . .

# Открываем порт 8501 (по умолчанию Streamlit использует этот порт)
EXPOSE 8501

# Команда для запуска Streamlit приложения
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
