import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import os
import pandas as pd
from light_mlt.light_mlt import fit_transform

def test_fit_transform_basic(tmp_path):
    df = pd.DataFrame({
        "city": ["Joinville", "Florianópolis", "Curitiba"],
        "age": [30, 25, 40]
    })
    outdir = tmp_path / "model_autoencoder"
    df_t, out_csv, token_cols, report = fit_transform(
        df, dir=str(outdir),
        categorical_cols=["city"], continuous_cols=["age"]
    )

    # Arquivos gerados
    assert (outdir / "schema.pkl").exists()
    assert (outdir / "mlt_params.pkl").exists()
    assert (outdir / "cats_map.pkl").exists()
    assert df_t.shape[0] == len(df)
    assert all(col in df_t.columns for col in token_cols)
