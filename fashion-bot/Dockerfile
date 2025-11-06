FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python","-m","app.main"]
