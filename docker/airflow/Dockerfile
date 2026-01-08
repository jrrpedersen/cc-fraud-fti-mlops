FROM apache/airflow:2.10.3

# 1) Install OS deps as root (Java for Spark)
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# 2) Install Python deps as airflow user (required by Airflow image)
USER airflow
RUN pip install --no-cache-dir \
    pyspark==3.5.3 \
    boto3==1.34.162 \
    requests==2.32.3 \
    redis==5.0.8 \
    pandas==2.2.3 \
    pyarrow==17.0.0 \
    scikit-learn==1.5.2 \
    mlflow==2.17.2