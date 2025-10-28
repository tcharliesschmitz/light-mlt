import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import pandas as pd
from light_mlt.light_mlt import fit_transform, inverse_transform

def test_inverse_transform_roundtrip(tmp_path):
    df = pd.DataFrame({
        "vehicle": ["Carreta", "Truck", "Cavalinho"],
        "experience": [10, 5, 8]
    })
    outdir = tmp_path / "mlt_test"
    df_t, _, _, _ = fit_transform(
        df, dir=str(outdir),
        categorical_cols=["vehicle"], continuous_cols=["experience"]
    )
    df_rec = inverse_transform(df_t, dir=str(outdir))

    # Comparação aproximada
    assert set(df_rec.columns) == set(df.columns)
    assert all(df_rec["vehicle"].isin(df["vehicle"].unique()))
    assert abs(df_rec["experience"].mean() - df["experience"].mean()) < 1e-3
