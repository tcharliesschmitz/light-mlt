import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import os
from light_mlt.light_mlt import _save_schema, _load_schema

def test_schema_io(tmp_path):
    schema = {"categorical_cols": ["city"], "continuous_cols": ["age"]}
    _save_schema(tmp_path, schema)
    schema2 = _load_schema(tmp_path)
    assert schema2 == schema
