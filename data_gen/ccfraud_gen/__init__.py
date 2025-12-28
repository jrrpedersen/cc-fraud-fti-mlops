"""Synthetic credit-card fraud data generator (v1).

Split into:
- bootstrap: generate slow-changing reference data (banks/merchants/accounts/cards)
- transactions: generate event stream (transactions) + labels (fraud_labels)

Designed for MinIO/S3 landing and future streaming (Kafka) scenarios.
"""
