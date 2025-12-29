from feast import Entity
from feast.value_type import ValueType

cc_num = Entity(
    name="cc_num",
    join_keys=["cc_num"],
    value_type=ValueType.STRING,
    description="Credit card number (entity id).",
)

merchant_id = Entity(
    name="merchant_id",
    join_keys=["merchant_id"],
    value_type=ValueType.STRING,
    description="Merchant identifier (entity id).",
)