# Dockerfile

# Stage 1: Copy dependencies from the dependencies image
FROM dms-model-dependencies AS dependencies

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
