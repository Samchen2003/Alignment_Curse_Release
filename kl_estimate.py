import torch
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def normalize_reps(Z_text, Z_audio, method="global_zscore", unit_norm=False, eps=1e-12):
    Zt = np.array(Z_text, dtype=np.float32, copy=True)
    Za = np.array(Z_audio, dtype=np.float32, copy=True)

    if method == "none":
        pass

    elif method == "global_zscore":
        allZ = np.vstack([Zt, Za])
        mu = allZ.mean(axis=0)
        std = allZ.std(axis=0, ddof=1)
        std = np.maximum(std, eps)
        Zt = (Zt - mu) / std
        Za = (Za - mu) / std

    elif method == "per_modality_zscore":
        mu_t = Zt.mean(axis=0); std_t = Zt.std(axis=0, ddof=1); std_t = np.maximum(std_t, eps)
        mu_a = Za.mean(axis=0); std_a = Za.std(axis=0, ddof=1); std_a = np.maximum(std_a, eps)
        Zt = (Zt - mu_t) / std_t
        Za = (Za - mu_a) / std_a

    elif method == "variance_match":
        var_t = np.maximum(Zt.var(axis=0, ddof=1), eps)
        var_a = np.maximum(Za.var(axis=0, ddof=1), eps)
        scale = np.sqrt(var_t / var_a)
        Za = (Za - Za.mean(axis=0)) * scale + Za.mean(axis=0)  
        mu = np.vstack([Zt, Za]).mean(axis=0)
        Zt = Zt - mu; Za = Za - mu

    elif method == "unit_norm":
        def l2norm_rows(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, eps)
            return X / norms
        Zt = l2norm_rows(Zt)
        Za = l2norm_rows(Za)

    elif method == "combined":
        Zt, Za = normalize_reps(Zt, Za, method="global_zscore", unit_norm=False, eps=eps)
        def l2norm_rows(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, eps)
            return X / norms
        Zt = l2norm_rows(Zt); Za = l2norm_rows(Za)

    else:
        raise ValueError(f"Unknown method: {method}")

    if unit_norm and method != "unit_norm" and method != "combined":
        norms_t = np.linalg.norm(Zt, axis=1, keepdims=True)
        norms_a = np.linalg.norm(Za, axis=1, keepdims=True)
        norms_t = np.maximum(norms_t, eps)
        norms_a = np.maximum(norms_a, eps)
        Zt = Zt / norms_t
        Za = Za / norms_a

    return Zt, Za
    
def load_json_entries(json_path):
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of entries {'prompt':..., 'audio_file':...}")
    return data


def pairwise_cosine_mean(Z1, Z2):
    def normalize(Z):
        nrm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return Z / nrm
    Z1n, Z2n = normalize(Z1), normalize(Z2)
    return float(np.mean(np.sum(Z1n * Z2n, axis=1)))




def classifier_density_ratio_kl(
    Z_audio,
    Z_text,
    *,
    calibrate: bool = True,
    n_splits: int = 5,
    C: float = 1.0,
    clip: float = 1e-6,
    max_iter: int = 2000,
    random_state: int = 0,
):
    Za = np.array(Z_audio)
    Zt = np.array(Z_text)
    X = np.vstack([Za, Zt])
    y = np.concatenate([np.ones(len(Za)), np.zeros(len(Zt))])

    if len(X) == 0:
        raise ValueError("Empty input arrays.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    log_r_values = []  

    base_clf_kwargs = {"penalty": "l2", "C": float(C), "solver": "lbfgs", "max_iter": int(max_iter)}

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        base = LogisticRegression(**base_clf_kwargs)

        if calibrate:
            clf = CalibratedClassifierCV(base, cv=3)
            clf.fit(X_train, y_train)
        else:
            base.fit(X_train, y_train)
            clf = base

        if np.any(y_test == 1):
            probs = clf.predict_proba(X_test[y_test == 1])[:, 1] 
            probs = np.clip(probs, clip, 1.0 - clip)
            log_r = np.log(probs / (1.0 - probs))
            log_r_values.append(log_r)
        else:
            continue

    if len(log_r_values) == 0:
        raise RuntimeError("No audio examples were present in held-out folds. Reduce n_splits or check data.")

    log_r_all = np.concatenate(log_r_values, axis=0)
    kl_est = float(np.mean(log_r_all))


    return kl_est

def run(
    Z_text,
    Z_audio,
    pca_dim=None,
    random_state=0,
):
    np.random.seed(random_state)


    print(f"Extracted shapes: text {Z_text.shape}, audio {Z_audio.shape}")

    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, whiten=True, svd_solver="randomized", random_state=random_state)
        pca.fit(np.vstack([Z_text, Z_audio]))
        Z_text = pca.transform(Z_text)
        Z_audio = pca.transform(Z_audio)
        print(f"PCA reduced to {pca_dim} dims")

    cos_mean = pairwise_cosine_mean(Z_audio, Z_text)

    results = {}

    kl_clf= classifier_density_ratio_kl(Z_audio, Z_text, calibrate=True, n_splits=5, C=1.0, clip=1e-6,random_state=random_state)

    results['kl_clf'] = float(kl_clf)
    print(f"Classifier-based KL estimate (audio||text): {kl_clf:.6f} ")

    return results





def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to embedding.npz")
    ap.add_argument("--pd", type=int, default=15, help="pca dimension")
    args = ap.parse_args(argv)

    Z_text = []
    Z_audio = []

    print("Loading embeddings...")
    data = np.load(args.path)
    Z_text = data["Z_text"]
    Z_audio = data["Z_audio"]
        
        
    Zt_norm, Za_norm = normalize_reps(
        Z_text, Z_audio, method="global_zscore", unit_norm=False
    )

    pd = args.pd
    res = run(Zt_norm, Za_norm, pca_dim=pd)
    
    print(float(res["kl_clf"]))

        
if __name__ == "__main__":
    main(sys.argv[1:])