FROM python:3.12-slim

# Dask offline store + redis online store
RUN pip install --no-cache-dir "feast[dask,redis]==0.58.0" pyarrow pandas

WORKDIR /feature_repo
CMD ["bash", "-lc", "feast --help"]
