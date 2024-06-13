# Dockerfile

# Stage 1: Copy dependencies from the dependencies image
FROM python:3.12-slim as dependencies

ENV http_proxy=deb.debian.org/debian
ENV https_proxy=deb.debian.org/debian

WORKDIR /app

COPY requirements-prod.txt /app

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxcb-cursor0
RUN pip install -r requirements-prod.txt


# Stage 2: Build the application
FROM python:3.12-slim as builder

ENV http_proxy=deb.debian.org/debian
ENV https_proxy=deb.debian.org/debian

COPY . /app
WORKDIR /app

COPY --from=dependencies / /

RUN rm -r /app/dms.py

# Stage 3: Final image
FROM python:3.12-slim

COPY --from=builder /app /app
COPY dms.py /app

CMD ["python", "/app/dms.py"]
