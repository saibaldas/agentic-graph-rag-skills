ARG PYTHON_TAG=3.11.17-slim
FROM python:${PYTHON_TAG}
ARG DEEPEVAL_VERSION=4.3.1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install "deepeval==${DEEPEVAL_VERSION}"
WORKDIR /app
COPY server.py /app/server.py
EXPOSE 8765
CMD ["python", "server.py"]
