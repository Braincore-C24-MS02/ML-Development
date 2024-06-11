FROM python:3.12-slim as builder

ENV http_proxy=deb.debian.org/debian
ENV https_proxy=deb.debian.org/debian

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r --default-timeout=100 requirements-prod.txt
RUN rm -r /app/dms.py

FROM python:3.12-slim

COPY --from=builder /app /app
COPY dms.py /app

CMD ["python", "/app/dms.py"]