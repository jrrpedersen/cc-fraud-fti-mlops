from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.data_format import ParquetFormat

from entities import cc_num, merchant_id

# NOTE:
# The Airflow DAG mirrors "current" snapshots to these local paths for Feast dev.
# This keeps Feast setup simple and avoids MinIO/S3 endpoint edge cases.
CARD_SOURCE = FileSource(
    name="card_features_current",
    path="data/offline/card_features/current",
    file_format=ParquetFormat(),
    timestamp_field="event_ts",
)

MERCHANT_SOURCE = FileSource(
    name="merchant_features_current",
    path="data/offline/merchant_features/current",
    file_format=ParquetFormat(),
    timestamp_field="event_ts",
)

card_txn_fv = FeatureView(
    name="card_txn_features",
    entities=[cc_num],
    ttl=timedelta(days=30),
    schema=[
        Field(name="cc_sum_amt_10m", dtype=Float32),
        Field(name="cc_cnt_10m", dtype=Int64),
        Field(name="cc_sum_amt_1h", dtype=Float32),
        Field(name="cc_cnt_1h", dtype=Int64),
        Field(name="cc_sum_amt_1d", dtype=Float32),
        Field(name="cc_cnt_1d", dtype=Int64),
        Field(name="cc_sum_amt_7d", dtype=Float32),
        Field(name="cc_cnt_7d", dtype=Int64),
        Field(name="cc_time_since_last_s", dtype=Int64),
    ],
    online=True,
    source=CARD_SOURCE,
)

merchant_risk_fv = FeatureView(
    name="merchant_risk_features",
    entities=[merchant_id],
    ttl=timedelta(days=90),
    schema=[
        Field(name="m_fraud_cnt_1d", dtype=Int64),
        Field(name="m_fraud_cnt_7d", dtype=Int64),
        Field(name="m_fraud_cnt_30d", dtype=Int64),
    ],
    online=True,
    source=MERCHANT_SOURCE,
)
