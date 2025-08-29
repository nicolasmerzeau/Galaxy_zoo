FROM python:3.10.6-slim

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY galaxy_zoo /galaxy_zoo

CMD uvicorn galaxy_zoo.api.fast:app --host 0.0.0.0 --port $PORT
