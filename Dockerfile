FROM python:3.10.9-slim

RUN pip install --no-cache-dir --upgrade pip


WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["sh", "startup.sh"]
