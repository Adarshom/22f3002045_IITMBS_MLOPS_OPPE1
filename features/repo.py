from feast import FeatureView, Entity, ValueType, Field
from feast.types import Float32, Int64, String
from data_source import file_source

stock = Entity(name="stock_name", value_type=ValueType.STRING, join_keys=["stock_name"])

schema = [
    Field(name="rolling_avg_10", dtype=Float32),
    Field(name="volume_sum_10", dtype=Float32),
    Field(name="close", dtype=Float32),
    Field(name="volume", dtype=Int64),
    Field(name="stock_name", dtype=String),
]

base_features = FeatureView(
    name="base_features",
    entities=[stock],
    ttl=None,
    schema=schema,
    source=file_source,
)
