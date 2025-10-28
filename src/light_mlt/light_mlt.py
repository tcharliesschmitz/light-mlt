# light_preprocessos.py
import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# Helpers de schema
# =========================
def _ensure_columns(df: pd.DataFrame,
                    categorical_cols: list[str],
                    continuous_cols: list[str],
                    unk_symbol: str = "∅") -> pd.DataFrame:
    """Garante que TODAS as colunas solicitadas existam no df, criando defaults se faltarem."""
    df = df.copy()
    for c in continuous_cols:
        if c not in df.columns:
            df[c] = 0.0
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = unk_symbol
    return df[categorical_cols + continuous_cols].copy()

def _safe_cast(df: pd.DataFrame, categorical_cols: list[str], continuous_cols: list[str]) -> pd.DataFrame:
    """Tipagem leve e robusta."""
    if continuous_cols:
        df[continuous_cols] = (
            df[continuous_cols]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float32")
            .fillna(0)
        )
    for c in categorical_cols:
        df[c] = df[c].astype("string").fillna("∅")
    return df

def _save_schema(dir: str, schema: dict) -> None:
    with open(os.path.join(dir, "schema.pkl"), "wb") as f:
        pickle.dump(schema, f)

def _load_schema(dir: str) -> dict:
    path = os.path.join(dir, "schema.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"schema.pkl não encontrado em {dir}. Rode fit() antes.")
    with open(path, "rb") as f:
        return pickle.load(f)

# ====== vocabulário mutável (append-only) ======
def _save_catmap(dir: str, cats_map: dict) -> None:
    with open(os.path.join(dir, "cats_map.pkl"), "wb") as f:
        pickle.dump(cats_map, f)

def _load_catmap(dir: str) -> dict:
    path = os.path.join(dir, "cats_map.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)
# =========================
# FIT (sem LabelEncoder)
# =========================
def fit(
    df: pd.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str],
    dir: str = "model_autoencoder/",
    unk_token: str = "__UNK__",
    mlt_p: int = 13,
    mlt_n: int = 6,
    mlt_config: dict | None = None,   # {"col":{"p":..,"n":..,"seed":..}}
    seed: int = 42,
):
    """
    Faz o fit inicial:
      - salva schema
      - cria e salva scaler (se houver contínuas)
      - cria e salva cats_map (ordenado + append do unk_token)
      - gera e salva parâmetros do MLT (p, n, M, Minv) por coluna categórica
      - RETORNA os nomes das novas colunas MLT por coluna e a lista total
    """
    os.makedirs(dir, exist_ok=True)

    schema = {
        "categorical_cols": list(categorical_cols or []),
        "continuous_cols":  list(continuous_cols or []),
        "unk_token":        unk_token,
        "unk_symbol":       "∅",
        "mlt_defaults":     {"p": int(mlt_p), "n": int(mlt_n), "seed": int(seed)},
        "mlt_config":       mlt_config or {},
        # será preenchido mais abaixo:
        "mlt_output_cols_by_cat": {},
        "mlt_output_cols_all": [],
    }
    _save_schema(dir, schema)

    df = _ensure_columns(df, schema["categorical_cols"], schema["continuous_cols"], schema["unk_symbol"])
    df = _safe_cast(df, schema["categorical_cols"], schema["continuous_cols"])

    # scaler
    if schema["continuous_cols"]:
        scaler = StandardScaler().fit(df[schema["continuous_cols"]].to_numpy(dtype=np.float64))
        with open(os.path.join(dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # cats_map (ordenado para determinismo)
    cats_map = {}
    n_classes = {}
    for c in schema["categorical_cols"]:
        labels = sorted(pd.Index(df[c].astype("string")).unique().tolist())
        if unk_token not in labels:
            labels.append(unk_token)  # reserva no fim
        cats_map[c] = {"labels": labels, "label2id": {lab: i for i, lab in enumerate(labels)}}
        n_classes[c] = len(labels)
    if cats_map:
        _save_catmap(dir, cats_map)

    # MLT
    mlt_cols_by_cat: dict[str, list[str]] = {}
    mlt_cols_all: list[str] = []

    if schema["categorical_cols"]:
        rng_global = np.random.default_rng(schema["mlt_defaults"]["seed"])
        mlt_params = {}

        for col in schema["categorical_cols"]:
            cfg = (schema["mlt_config"] or {}).get(col, {})
            p_c   = int(cfg.get("p",    schema["mlt_defaults"]["p"]))
            n_c   = int(cfg.get("n",    schema["mlt_defaults"]["n"]))
            seed_c= int(cfg.get("seed", schema["mlt_defaults"]["seed"]))

            rng = np.random.default_rng(seed_c) if col in (schema["mlt_config"] or {}) else rng_global

            if (p_c ** n_c) < max(1, n_classes.get(col, 1)):
                raise ValueError(f"[{col}] Capacidade insuficiente: p^n={p_c}^{n_c} < {n_classes.get(col, 1)} classes.")

            # Matriz inversível n_c x n_c (mod p_c)
            M    = random_invertible_matrix_mod_p(n_c, p_c, rng).astype(int)
            Minv = gauss_jordan_inv_mod_p(M, p_c).astype(int)

            mlt_params[col] = {"p": p_c, "n": n_c, "M": M, "Minv": Minv}

            # >>> nomes das novas colunas para esta coluna categórica <<<
            # Padrão: {col}__mlt_{i}, i=0..n_c-1
            out_cols = [f"{col}_t{i}" for i in range(n_c)]
            mlt_cols_by_cat[col] = out_cols
            mlt_cols_all.extend(out_cols)

        # salva params
        with open(os.path.join(dir, "mlt_params.pkl"), "wb") as f:
            pickle.dump(mlt_params, f)

    # salvar no schema os nomes das saídas do MLT
    schema["mlt_output_cols_by_cat"] = mlt_cols_by_cat
    schema["mlt_output_cols_all"] = mlt_cols_all

    # construir dtypes no formato pandas
    dtypes = {}
    for col in schema["continuous_cols"]:
        dtypes[col] = pd.Series([], dtype="float32").dtype
    for col in mlt_cols_all:
        dtypes[col] = pd.Series([], dtype="int64").dtype   # ou float32 se tokens forem floats

    schema["output_dtypes"] = {k: str(v) for k, v in dtypes.items()}
    _save_schema(dir, schema)

    # >>> retorno solicitado <<<
    return pd.Series(dtypes)


# =========================
# TRANSFORM (usa cats_map + scaler + MLT)
# =========================
def transform(
    df: pd.DataFrame,
    dir: str = "model_autoencoder/",
    output_csv: str | None = None,
    drop_original: bool = True,
):
    """
    Retorna:
      - df_t: DataFrame transformado (contínuas escaladas + tokens MLT)
      - cat_token_cols: lista com os nomes das NOVAS colunas categóricas (tokens MLT)
      - cont_cols_new: lista com os nomes das NOVAS colunas contínuas (mesmos nomes, porém escalados)
    """
    schema     = _load_schema(dir)
    cat_cols   = schema.get("categorical_cols", [])
    cont_cols  = schema.get("continuous_cols", [])
    unk_token  = schema.get("unk_token", "__UNK__")
    unk_symbol = schema.get("unk_symbol", "-999")

    # sanity + tipos
    df   = _ensure_columns(df, cat_cols, cont_cols, unk_symbol)
    df   = _safe_cast(df, cat_cols, cont_cols)

    # contínuas (escala se houver scaler)
    if cont_cols:
        scaler = None
        spath  = os.path.join(dir, "scaler.pkl")
        if os.path.exists(spath):
            with open(spath, "rb") as f:
                scaler = pickle.load(f)
        Xc = (scaler.transform(df[cont_cols].to_numpy(dtype=np.float64))
              if scaler is not None else df[cont_cols].to_numpy())
        cont_df = pd.DataFrame(Xc, columns=cont_cols, index=df.index, dtype="float32")
    else:
        cont_df = pd.DataFrame(index=df.index)

    # categóricas -> ids (sem crescimento)
    if cat_cols:
        cats_map = _load_catmap(dir)
        cats = {}
        for c in cat_cols:
            cm        = cats_map.get(c, {"labels":[unk_token], "label2id":{unk_token:0}})
            label2id  = cm["label2id"]
            unk_idx   = label2id.get(unk_token, max(label2id.values()) if label2id else 0)
            cats[c]   = df[c].map(label2id).fillna(unk_idx).astype("int64")
        cat_df = pd.DataFrame(cats, index=df.index)
    else:
        cat_df = pd.DataFrame(index=df.index)

    # aplica MLT
    df_t = pd.concat([cont_df, cat_df], axis=1)
    cat_token_cols: list[str] = []
    if cat_cols:
        df_t, cat_token_cols = apply_mlt_tokens(df_t, cat_cols, dir=dir, drop_original=drop_original)

    # limpeza numérica
    df_t.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_t.fillna(0, inplace=True)

    # opcional: salvar
    if output_csv:
        df_t.to_csv(output_csv, index=False)

    # novas contínuas = mesmas colunas contínuas (apenas escaladas)
    cont_cols_new = list(cont_df.columns)

    return df_t, cat_token_cols, cont_cols_new


# =========================
# INVERSE TRANSFORM (usa só cats_map)
# =========================
def inverse_transform(
    df_t: pd.DataFrame,
    dir: str = "model_autoencoder/",
) -> pd.DataFrame:
    """
    Reconstrói o DataFrame original (categóricas e contínuas) a partir do transformado,
    usando apenas cats_map, mlt_params e scaler.
    """
    schema = _load_schema(dir)
    cat_cols = schema["categorical_cols"]
    con_cols = schema["continuous_cols"]
    unk_token = schema["unk_token"]

    df_rec = pd.DataFrame(index=df_t.index)
    cats_map = _load_catmap(dir)

    # Categóricas: MLT^-1 -> ids -> labels
    if cat_cols:
        with open(os.path.join(dir, "mlt_params.pkl"), "rb") as f:
            mp = pickle.load(f)
        for c in cat_cols:
            n = mp[c]["n"]; tcols = [f"{c}_t{i}" for i in range(n)]
            if not all(tc in df_t.columns for tc in tcols):
                df_rec[c] = unk_token; continue
            ids = mlt_decode_tokens_for_column(df_t, c, mp)
            labels = cats_map.get(c, {}).get("labels", [])
            out = [labels[i] if (0 <= i < len(labels)) else unk_token for i in ids]
            df_rec[c] = pd.Series(out, index=df_t.index)

    # Contínuas
    scaler_path = os.path.join(dir, "scaler.pkl")
    if con_cols and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.inverse_transform(df_t[con_cols].to_numpy(dtype=np.float64, copy=False))
        df_rec[con_cols] = pd.DataFrame(X, index=df_t.index, columns=con_cols).astype("float32")
    elif con_cols:
        for c in con_cols:
            df_rec[c] = pd.to_numeric(df_t.get(c, 0), errors="coerce").fillna(0).astype("float32")

    return df_rec

# =========================
# FIT_TRANSFORM (crescimento seguro do cats_map)
# =========================
def fit_transform(
    df: pd.DataFrame,
    dir: str = "model_autoencoder/",
    # Só precisa passar as listas na 1ª vez; depois vêm do schema salvo
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
    drop_original: bool = True,
    tokens_as_float: bool = True,     # útil p/ FAISS
    on_capacity: str = "error",       # "error" | "ignore" (não recomendável)
):
    """
    Faz fit (se necessário) e transform.
    - Se artefatos já existem, transforma e ANEXA novas categorias ao cats_map (append-only).
    - Mantém MLT (p,n,M) fixos; se ainda não existir, cria com capacidade >= #labels atuais.
    Retorna: (df_t, output_csv, token_cols, report)
    """
    os.makedirs(dir, exist_ok=True)
    schema_path = os.path.join(dir, "schema.pkl")
    mlt_path    = os.path.join(dir, "mlt_params.pkl")
    scaler_path = os.path.join(dir, "scaler.pkl")

    # schema
    if os.path.exists(schema_path):
        schema = _load_schema(dir)
    else:
        schema = {
            "categorical_cols": list(categorical_cols or []),
            "continuous_cols":  list(continuous_cols or []),
            "unk_token":        "__UNK__",
            "unk_symbol":       "∅",
            "mlt_defaults":     {"p": 13, "n": 6, "seed": 42},
            "mlt_config":       {},
        }
        _save_schema(dir, schema)

    cat_cols = schema["categorical_cols"]
    con_cols = schema["continuous_cols"]
    unk_token = schema["unk_token"]
    unk_symbol = schema["unk_symbol"]

    df = _ensure_columns(df, cat_cols, con_cols, unk_symbol)
    df = _safe_cast(df, cat_cols, con_cols)

    # scaler
    scaler = None
    if con_cols:
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        else:
            scaler = StandardScaler().fit(df[con_cols].to_numpy(dtype=np.float64))
            with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    # cats_map: cria se não existir (ordenado), depois permite crescimento
    cats_map = _load_catmap(dir)
    if not cats_map:
        for c in cat_cols:
            labels = sorted(pd.Index(df[c].astype("string")).unique().tolist())
            if unk_token not in labels: labels.append(unk_token)
            cats_map[c] = {"labels": labels, "label2id": {lab: i for i, lab in enumerate(labels)}}
        _save_catmap(dir, cats_map)

    cats_map_before = {k: v["labels"][:] for k, v in cats_map.items()}
    changed = False

    # crescimento (append-only)
    for c in cat_cols:
        labels   = cats_map[c]["labels"]
        label2id = cats_map[c]["label2id"]
        cap = float("inf")
        if os.path.exists(mlt_path):
            with open(mlt_path, "rb") as f:
                mp = pickle.load(f); p = mp[c]["p"]; n = mp[c]["n"]; cap = p ** n
        new_vals = pd.Index(df[c].astype("string").unique())
        for v in new_vals:
            if v not in label2id:
                next_id = len(labels)
                if next_id >= cap:
                    msg = f"[{c}] Sem capacidade MLT: {next_id} >= p^n={cap}. Refaça fit com p/n maiores."
                    if on_capacity == "error": raise ValueError(msg)
                    else: continue
                labels.append(v); label2id[v] = next_id; changed = True
    if changed:
        _save_catmap(dir, cats_map)

    # cria MLT se não existir (capacidade pelo tamanho do cats_map)
    if cat_cols and (not os.path.exists(mlt_path)):
        rng_global = np.random.default_rng(schema["mlt_defaults"]["seed"])
        mlt_params = {}
        for c in cat_cols:
            cfg = (schema["mlt_config"] or {}).get(c, {})
            p_c = int(cfg.get("p", schema["mlt_defaults"]["p"]))
            n_c = int(cfg.get("n", schema["mlt_defaults"]["n"]))
            seed_c = int(cfg.get("seed", schema["mlt_defaults"]["seed"]))
            rng = np.random.default_rng(seed_c) if c in (schema["mlt_config"] or {}) else rng_global
            n_classes = len(cats_map[c]["labels"])
            if p_c ** n_c < n_classes:
                raise ValueError(f"[{c}] Capacidade insuficiente: p^n={p_c}^{n_c} < {n_classes}.")
            M = random_invertible_matrix_mod_p(n_c, p_c, rng).astype(int)
            Minv = gauss_jordan_inv_mod_p(M, p_c).astype(int)
            mlt_params[c] = {"p": p_c, "n": n_c, "M": M, "Minv": Minv}
        with open(mlt_path, "wb") as f:
            pickle.dump(mlt_params, f)

    # mapping via cats_map (sem UNK, porque já anexamos novas)
    if cat_cols:
        cats = {c: df[c].map(cats_map[c]["label2id"]).astype("int64") for c in cat_cols}
        cats = pd.DataFrame(cats, index=df.index)
    else:
        cats = pd.DataFrame(index=df.index)

    # contínuas
    if con_cols:
        cont = pd.DataFrame(
            scaler.transform(df[con_cols].to_numpy(dtype=np.float64)) if scaler is not None else df[con_cols].to_numpy(),
            columns=con_cols, index=df.index, dtype="float32"
        )
    else:
        cont = pd.DataFrame(index=df.index)

    # MLT
    df_t = pd.concat([cont, cats], axis=1)
    token_cols = []
    if cat_cols:
        df_t, token_cols = apply_mlt_tokens(df_t, cat_cols, dir=dir, drop_original=drop_original)
        if tokens_as_float:
            df_t[token_cols] = df_t[token_cols].astype("float32")

    df_t.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_t.fillna(0, inplace=True)
    out = os.path.join(dir, "preprocessed.csv")
    if os.path.exists(out): os.remove(out)
    df_t.to_csv(out, index=False)

    # relatório: headroom e categorias adicionadas
    with open(mlt_path, "rb") as f: mp = pickle.load(f)
    cats_map_now = _load_catmap(dir)
    headroom, added = {}, {}
    for c in cat_cols:
        cap = int(mp[c]["p"] ** mp[c]["n"])
        seen = len(cats_map_now[c]["labels"])
        headroom[c] = cap - seen
        before = set(cats_map_before.get(c, []))
        now    = set(cats_map_now.get(c, {}).get("labels", []))
        added[c] = sorted(list(now - before))

    return df_t, out, token_cols, {"added": added, "headroom": headroom}

# ==========================================================
# -------------------  MLT (base-p vetorizado) ------------
# ==========================================================
def to_base_p_matrix(ids: np.ndarray, p: int, n: int) -> np.ndarray:
    """Converte vetor de IDs (>=0) para matriz de dígitos base-p (B, n)."""
    ids = np.asarray(ids, dtype=np.int64)
    if np.any(ids < 0):
        raise ValueError("IDs devem ser não-negativos.")
    D = np.empty((ids.shape[0], n), dtype=np.int64)
    x = ids.copy()
    for i in range(n-1, -1, -1):
        x, D[:, i] = np.divmod(x, p)
    if np.any(x != 0):
        raise ValueError("Algum ID não cabe em n dígitos base p.")
    return D

def from_base_p_matrix(D: np.ndarray, p: int) -> np.ndarray:
    """Converte matriz de dígitos (B, n) em inteiros (vetorizado)."""
    D = np.asarray(D, dtype=np.int64)
    vals = np.zeros(D.shape[0], dtype=np.int64)
    for i in range(D.shape[1]):
        vals = vals * p + D[:, i]
    return vals

def gauss_jordan_inv_mod_p(M: np.ndarray, p: int) -> np.ndarray:
    n = M.shape[0]
    A = (M % p).astype(int)
    I = np.eye(n, dtype=int)
    A = np.concatenate([A, I], axis=1) % p
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if A[r, col] % p != 0:
                pivot = r; break
        if pivot is None:
            raise ValueError("M não é invertível em Z_p (det ≡ 0).")
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        inv_pivot = pow(int(A[row, col]) % p, -1, p)
        A[row, :] = (A[row, :] * inv_pivot) % p
        for r in range(n):
            if r != row and A[r, col] % p != 0:
                factor = A[r, col] % p
                A[r, :] = (A[r, :] - factor * A[row, :]) % p
        row += 1
    return A[:, n:] % p

def random_invertible_matrix_mod_p(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        M = rng.integers(0, p, size=(n, n), dtype=np.int64)
        try:
            _ = gauss_jordan_inv_mod_p(M, p)
            return M
        except ValueError:
            continue

def apply_mlt_tokens(
    df_processed: pd.DataFrame,
    categorical_cols: list[str],
    dir: str = "model_autoencoder/",
    p: int | None = None,
    n: int | None = None,
    mlt_config: dict | None = None,
    seed: int = 42,
    drop_original: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Expande colunas categóricas inteiras em n tokens MLT em Z_p.
    PRIORIDADE:
      1) Se existir 'mlt_params.pkl' no dir, usa (p, n, M) por coluna.
      2) Caso contrário, gera com base no schema/cats_map (capacidade >= #labels) e salva.
    """
    if len(categorical_cols) == 0:
        return df_processed.copy(), []

    os.makedirs(dir, exist_ok=True)
    params_path = os.path.join(dir, "mlt_params.pkl")
    use_saved = os.path.exists(params_path)

    if use_saved:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    else:
        schema = _load_schema(dir)
        rng_global = np.random.default_rng(schema["mlt_defaults"]["seed"])
        params = {}

    out = df_processed.copy()
    novas_colunas = []

    # Para sizing inicial (quando não há mlt_params), use #labels do cats_map
    cats_map = _load_catmap(dir)

    for col in categorical_cols:
        if use_saved and col in params:
            p_c = int(params[col]["p"])
            n_c = int(params[col]["n"])
            M   = np.array(params[col]["M"], dtype=int)
        else:
            schema = _load_schema(dir)
            cfg = (schema["mlt_config"] or {}).get(col, {})
            p_c = int(cfg.get("p", schema["mlt_defaults"]["p"] if p is None else p))
            n_c = int(cfg.get("n", schema["mlt_defaults"]["n"] if n is None else n))
            seed_c = int(cfg.get("seed", schema["mlt_defaults"]["seed"]))
            rng = np.random.default_rng(seed_c)

            # capacidade pela cardinalidade do vocabulário da coluna
            n_classes = len(cats_map.get(col, {}).get("labels", []))
            if p_c ** n_c < max(1, n_classes):
                raise ValueError(f"[{col}] Capacidade insuficiente: p^n={p_c}^{n_c} < {n_classes} classes.")
            M = random_invertible_matrix_mod_p(n_c, p_c, rng).astype(int)
            Minv = gauss_jordan_inv_mod_p(M, p_c).astype(int)
            params[col] = {"p": p_c, "n": n_c, "M": M, "Minv": Minv}

        # ids inteiros; se coluna não existir, usa zeros
        if col in out.columns:
            ids = out[col].to_numpy(dtype=np.int64, copy=False)
        else:
            ids = np.zeros(len(out), dtype=np.int64)

        D = to_base_p_matrix(ids, p_c, n_c)        # (B, n)
        Y = (D @ (M % p_c)) % p_c                  # (B, n)

        token_cols = [f"{col}_t{i}" for i in range(n_c)]
        out[token_cols] = pd.DataFrame(Y, index=out.index)
        novas_colunas.extend(token_cols)

        if drop_original and col in out.columns:
            out.drop(columns=[col], inplace=True)

    if not use_saved:
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

    return out, novas_colunas

def mlt_decode_tokens_for_column(
    df_tokens: pd.DataFrame, col: str, params: dict
) -> np.ndarray:
    """Decodifica tokens MLT para IDs inteiros usando Minv, com base-p vetorizada."""
    p = params[col]["p"]
    n = params[col]["n"]
    Minv = params[col]["Minv"]
    token_cols = [f"{col}_t{i}" for i in range(n)]
    T = np.zeros((len(df_tokens), n), dtype=np.int64)
    for i, tcol in enumerate(token_cols):
        if tcol in df_tokens.columns:
            T[:, i] = df_tokens[tcol].to_numpy(dtype=np.int64) % p
    D = (T @ Minv) % p
    ids = from_base_p_matrix(D, p)
    return ids

def load_mlt_params(dir: str = "model_autoencoder/") -> dict:
    with open(os.path.join(dir, "mlt_params.pkl"), "rb") as f:
        return pickle.load(f)
