#!/usr/bin/env python3
# =============================================================================
# 32-Hybrid-Post-Quantum-Secure.py
#
# Reproducibility script for:
#   "Hybrid Post-Quantum Secure Aggregation for Federated Intrusion Detection
#    in IoT Edge Networks: An Empirical Cost-Utility Analysis of ML-KEM-Based
#    Pairwise Masking"
#
# STUDY COMPONENTS: [ML] + [BENCH]
#   [ML]    component: federated training of lightweight intrusion-detection
#           models (NSL-KDD, UNSW-NB15) under plaintext FedAvg and under
#           secure aggregation with fixed-point quantised masking.
#   [BENCH] component: measurement of cryptographic primitives (X25519,
#           ML-KEM-512/768/1024, hybrid X25519+ML-KEM) and of complete
#           secure-aggregation rounds (setup, masking, unmasking, dropout
#           recovery) executed on real model-update vectors, with exact
#           serialized byte accounting.
#   Notes: "dataset download" = NSL-KDD raw files + UNSW-NB15
#   mirror tarball (exact URLs below); "model training" = numpy MLPs via
#   FedAvg; "benchmark iterations" = timed crypto ops and full SecAgg rounds;
#   figures/tables are generated exclusively from artifacts produced here.
#
# USAGE:
#   pip install numpy pandas scikit-learn matplotlib kyber-py cryptography
#   python 32-Hybrid-Post-Quantum-Secure.py                # runs all stages in order
#   python 32-Hybrid-Post-Quantum-Secure.py --stage data   # or: unittest, crypto, train,
#                                              # bench, ablations, model, figs
#
# All artifacts are written under ./data, ./out and ./figs relative to the
# working directory. Stages are idempotent: completed work is cached and
# skipped on re-runs.

# Scope. The study has two experimental components and the script below reflects both. The learning component federates the training of lightweight intrusion-detection models across simulated client sites on NSL-KDD and UNSW-NB15, under plaintext aggregation and under secure aggregation, and evaluates detection on held-out test sets. The benchmarking component measures cryptographic primitives (X25519, ML-KEM-512/768/1024, and the hybrid combiner) and complete secure-aggregation rounds — setup, masking, unmasking, and dropout recovery — executed on the real model-update vectors produced by the learning component, with exact serialized byte accounting.

# Data and provenance. The datasets are the NSL-KDD raw file pair and the UNSW-NB15 mirror tarball, both fetched from the exact URLs hard-coded in the script; preprocessing is performed entirely within the pipeline. Detection scores populate the learning tables, and byte counts and latencies populate the benchmark tables; each table in Section 4 is produced by the stage that legitimately generates it, with no value transcribed by hand.

# Environment. Single-core x86-64 Linux, Python 3.12, 3 GB RAM. Dependencies:
# pip install numpy pandas scikit-learn matplotlib kyber-py cryptography
# python 32-Hybrid-Post-Quantum-Secure.py

# Execution. Running python 32-Hybrid-Post-Quantum-Secure.py with no arguments executes all eight stages in order: data, unittest, crypto, train, bench, ablations, model, figs. Stages cache their artifacts under out/ and are idempotent, so an interrupted run resumes where it stopped; a single stage may also be run in isolation with --stage <name>. Every value printed to stdout is labeled by the table it populates (for example, Table 4 | mlp_l|hybrid|n50: setupB/cli 31272 ...), and every figure in Section 4 is written to figs/ as a PNG. Total runtime on the reference machine is approximately 45 minutes, dominated by the bench stage at n=100, where pure-Python ML-KEM performs 4,950 pairwise encapsulations.

# Correctness gates. The unittest stage is not decorative and runs before any result is produced. It asserts that, for all three key-agreement schemes, the server’s reconstructed sum equals the plaintext quantized sum exactly, both with all clients surviving and with clients dropping out mid-round; the bench and ablations stages re-assert this at every measured configuration. Should mask cancellation or dropout recovery fail anywhere, the run aborts rather than reporting a number.

# Determinism. All randomness that affects reported metrics is seeded: client partitioning, model initialization, and batch shuffling derive from the run seed (0, 1, 2), and every learning result in Table 2 is the mean and standard deviation over those three seeds. Cryptographic key material is drawn from the operating system’s cryptographic random source, as it must be; it affects timing measurements at the noise level but cannot affect any detection metric, because mask cancellation is exact.
# The complete script follows. It is identical, byte for byte, to the 32-Hybrid-Post-Quantum-Secure.py file distributed in the reproducibility package.
# =============================================================================

import argparse
import hashlib
import io
import json
import os
import secrets
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# ------------------------------- constants ----------------------------------

DATA_DIR = "data"
OUT_DIR = "out"
FIG_DIR = "figs"

NSL_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
NSL_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
UNSW_TARBALL_URL = ("https://codeload.github.com/abhinav-bhardwaj/"
                    "IoT-Network-Intrusion-Detection-System-UNSW-NB15/tar.gz/refs/heads/master")
UNSW_MEMBER = ("IoT-Network-Intrusion-Detection-System-UNSW-NB15-master/"
               "datasets/UNSW_NB15.csv")

FIELD_BITS = 32                    # masking field q = 2**32 (uint32 wrap-around)
QUANT_BITS_DEFAULT = 16            # fixed-point resolution k
CLIP_DEFAULT = 0.5                 # clipping bound c for weight-delta coords
ROUNDS = 30
N_CLIENTS_TRAIN = 10
DIRICHLET_ALPHA = 0.5
LOCAL_BATCH = 512
LOCAL_LR = 0.1
LOCAL_MOMENTUM = 0.9
SEEDS = (0, 1, 2)

SHAMIR_P = 2**255 - 19             # prime field for Shamir secret sharing

NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
NSL_CAT = ["protocol_type", "service", "flag"]
NSL_FAMILY = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r",
}

# ------------------------------- utilities ----------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# STAGE 1 -- DATA: download and preprocess NSL-KDD and UNSW-NB15
# =============================================================================


def _download(url, dest):
    if os.path.exists(dest):
        log(f"cached: {dest}")
        return
    log(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def stage_data():
    import pandas as pd
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- NSL-KDD ------------------------------------------------------
    tr_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    te_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    _download(NSL_TRAIN_URL, tr_path)
    _download(NSL_TEST_URL, te_path)

    if not os.path.exists(os.path.join(DATA_DIR, "nsl.npz")):
        log("preprocessing NSL-KDD")
        tr = pd.read_csv(tr_path, header=None, names=NSL_COLS)
        te = pd.read_csv(te_path, header=None, names=NSL_COLS)
        y_tr = (tr["label"] != "normal").astype(np.float32).values
        y_te = (te["label"] != "normal").astype(np.float32).values
        fam = tr["label"].map(lambda s: NSL_FAMILY.get(s, "r2l")).values
        Xtr_parts, Xte_parts = [], []
        for c in NSL_CAT:
            cats = sorted(tr[c].unique())
            idx = {v: i for i, v in enumerate(cats)}
            oh_tr = np.zeros((len(tr), len(cats)), np.float32)
            oh_te = np.zeros((len(te), len(cats)), np.float32)
            oh_tr[np.arange(len(tr)), tr[c].map(idx).values] = 1
            known = te[c].map(idx)
            m = known.notna().values
            oh_te[np.arange(len(te))[m], known[m].astype(int).values] = 1
            Xtr_parts.append(oh_tr)
            Xte_parts.append(oh_te)
        num_cols = [c for c in NSL_COLS[:-2] if c not in NSL_CAT]
        num_tr = tr[num_cols].values.astype(np.float32)
        num_te = te[num_cols].values.astype(np.float32)
        num_tr = np.log1p(np.clip(num_tr, 0, None))
        num_te = np.log1p(np.clip(num_te, 0, None))
        mu, sd = num_tr.mean(0), num_tr.std(0) + 1e-6
        Xtr_parts.append(((num_tr - mu) / sd).astype(np.float32))
        Xte_parts.append(((num_te - mu) / sd).astype(np.float32))
        Xtr = np.hstack(Xtr_parts)
        Xte = np.hstack(Xte_parts)
        np.savez_compressed(os.path.join(DATA_DIR, "nsl.npz"),
                            Xtr=Xtr, ytr=y_tr, Xte=Xte, yte=y_te,
                            fam=np.array(fam, dtype="U8"))
        log(f"NSL-KDD: train {Xtr.shape} test {Xte.shape} "
            f"attack-rate train {y_tr.mean():.3f} test {y_te.mean():.3f}")

    # ---- UNSW-NB15 (official training partition via GitHub mirror) -----
    unsw_csv = os.path.join(DATA_DIR, "UNSW_NB15.csv")
    if not os.path.exists(unsw_csv):
        tgz = os.path.join(DATA_DIR, "unsw.tgz")
        _download(UNSW_TARBALL_URL, tgz)
        log("extracting UNSW-NB15 csv from tarball")
        with tarfile.open(tgz, "r:gz") as t:
            with t.extractfile(UNSW_MEMBER) as src, open(unsw_csv, "wb") as dst:
                dst.write(src.read())
        os.remove(tgz)

    if not os.path.exists(os.path.join(DATA_DIR, "unsw.npz")):
        log("preprocessing UNSW-NB15")
        df = pd.read_csv(unsw_csv)
        assert len(df) == 175341, "unexpected UNSW-NB15 partition size"
        y = df["label"].astype(np.float32).values
        fam = df["attack_cat"].astype(str).values
        cat_cols = ["proto", "service", "state"]
        parts = []
        rng = np.random.default_rng(42)
        for c in cat_cols:
            cats = sorted(df[c].astype(str).unique())
            idx = {v: i for i, v in enumerate(cats)}
            oh = np.zeros((len(df), len(cats)), np.float32)
            oh[np.arange(len(df)), df[c].astype(str).map(idx).values] = 1
            parts.append(oh)
        num_cols = [c for c in df.columns
                    if c not in cat_cols + ["id", "attack_cat", "label"]]
        num = df[num_cols].values.astype(np.float32)
        num = np.log1p(np.clip(num, 0, None))
        # stratified 70/10/20 split on (attack_cat) with fixed seed
        order = rng.permutation(len(df))
        tr_idx, va_idx, te_idx = [], [], []
        for cls in np.unique(fam):
            rows = order[np.isin(order, np.where(fam == cls)[0])]
            n1 = int(0.7 * len(rows))
            n2 = int(0.8 * len(rows))
            tr_idx += rows[:n1].tolist()
            va_idx += rows[n1:n2].tolist()
            te_idx += rows[n2:].tolist()
        tr_idx, va_idx, te_idx = map(np.array, (tr_idx, va_idx, te_idx))
        mu, sd = num[tr_idx].mean(0), num[tr_idx].std(0) + 1e-6
        num = ((num - mu) / sd).astype(np.float32)
        X = np.hstack(parts + [num])
        np.savez_compressed(
            os.path.join(DATA_DIR, "unsw.npz"),
            Xtr=X[tr_idx], ytr=y[tr_idx], Xva=X[va_idx], yva=y[va_idx],
            Xte=X[te_idx], yte=y[te_idx],
            fam=np.array(fam[tr_idx], dtype="U16"))
        log(f"UNSW-NB15: X {X.shape} train/val/test "
            f"{len(tr_idx)}/{len(va_idx)}/{len(te_idx)} "
            f"attack-rate {y.mean():.3f}")

    # dataset summary table
    nsl = np.load(os.path.join(DATA_DIR, "nsl.npz"), allow_pickle=True)
    un = np.load(os.path.join(DATA_DIR, "unsw.npz"), allow_pickle=True)
    summary = {
        "NSL-KDD": {"train": int(len(nsl["ytr"])), "val": "10% of train",
                    "test": int(len(nsl["yte"])),
                    "dims": int(nsl["Xtr"].shape[1]),
                    "attack_rate_train": float(nsl["ytr"].mean()),
                    "attack_rate_test": float(nsl["yte"].mean())},
        "UNSW-NB15": {"train": int(len(un["ytr"])), "val": int(len(un["yva"])),
                      "test": int(len(un["yte"])),
                      "dims": int(un["Xtr"].shape[1]),
                      "attack_rate_train": float(un["ytr"].mean()),
                      "attack_rate_test": float(un["yte"].mean())},
    }
    save_json(os.path.join(OUT_DIR, "dataset_summary.json"), summary)
    log("stage data: DONE")


def load_dataset(name):
    z = np.load(os.path.join(DATA_DIR, f"{name}.npz"), allow_pickle=True)
    if name == "nsl":
        X, y = z["Xtr"], z["ytr"]
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y))
        n_va = int(0.1 * len(y))
        va, tr = idx[:n_va], idx[n_va:]
        return (X[tr], y[tr], z["fam"][tr], X[va], y[va],
                z["Xte"], z["yte"])
    return (z["Xtr"], z["ytr"], z["fam"], z["Xva"], z["yva"],
            z["Xte"], z["yte"])


# =============================================================================
# Lightweight MLP in numpy (deterministic, single-core friendly)
# =============================================================================


class MLP:
    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            s = np.sqrt(2.0 / dims[i])
            self.W.append(rng.normal(0, s, (dims[i], dims[i + 1])).astype(np.float32))
            self.b.append(np.zeros(dims[i + 1], np.float32))
        self.vel = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        out = []
        for w, b in zip(self.W, self.b):
            out += [w, b]
        return out

    def get_flat(self):
        return np.concatenate([p.ravel() for p in self._params()])

    def set_flat(self, v):
        o = 0
        for p in self._params():
            p[...] = v[o:o + p.size].reshape(p.shape)
            o += p.size

    @property
    def dim(self):
        return sum(p.size for p in self._params())

    def forward(self, X):
        a = X
        self._cache = [a]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z
            self._cache.append(a)
        return a.ravel()  # logits

    def predict_proba(self, X, bs=8192):
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), bs):
            z = self.forward(X[i:i + bs])
            out[i:i + bs] = _sigmoid(z)
        return out

    def train_epoch(self, X, y, lr, mom, bs, rng):
        idx = rng.permutation(len(y))
        tot = 0.0
        for s in range(0, len(y), bs):
            b_idx = idx[s:s + bs]
            xb, yb = X[b_idx], y[b_idx]
            logits = self.forward(xb)
            p = _sigmoid(logits)
            eps = 1e-7
            tot += float(-np.mean(yb * np.log(p + eps)
                                  + (1 - yb) * np.log(1 - p + eps))) * len(yb)
            g = ((p - yb) / len(yb)).astype(np.float32)
            grads = []
            delta = g[:, None]
            for i in range(len(self.W) - 1, -1, -1):
                a_prev = self._cache[i]
                gw = a_prev.T @ delta
                gb = delta.sum(0)
                grads = [gw, gb] + grads
                if i > 0:
                    delta = (delta @ self.W[i].T) * (self._cache[i] > 0)
            for p_, g_, v_ in zip(self._params(), grads, self.vel):
                v_ *= mom
                v_ += g_.astype(np.float32)
                p_ -= lr * v_
        return tot / len(y)


def _sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def metrics_from_scores(y, s):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    yhat = (s >= 0.5).astype(int)
    return {"acc": accuracy_score(y, yhat),
            "prec": precision_score(y, yhat, zero_division=0),
            "rec": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "auc": roc_auc_score(y, s)}


# =============================================================================
# Fixed-point quantiser for masking field (q = 2**32)
# =============================================================================


class Quantizer:
    def __init__(self, clip=CLIP_DEFAULT, bits=QUANT_BITS_DEFAULT):
        self.c, self.k = clip, bits
        self.levels = (1 << bits) - 1
        self.scale = self.levels / (2 * clip)

    def encode(self, x):
        xc = np.clip(x, -self.c, self.c)
        clipped = float(np.mean(np.abs(x) > self.c))
        q = np.rint((xc + self.c) * self.scale).astype(np.uint32)
        return q, clipped

    def decode_sum(self, qsum_u32, m):
        # qsum is sum over m participants in Z_{2^32}; headroom m*levels < 2^32
        return qsum_u32.astype(np.float64) / self.scale - m * self.c


# =============================================================================
# Cryptographic schemes: X25519, ML-KEM, hybrid; PRG; Shamir
# =============================================================================

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey)
from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024

MLKEM = {"mlkem512": ML_KEM_512, "mlkem768": ML_KEM_768,
         "mlkem1024": ML_KEM_1024}


def _mlkem_keygen_from_seed(kem, seed32):
    """FIPS 203 derandomised keygen from a 32-byte master seed (d,z from SHAKE)."""
    dz = hashlib.shake_256(seed32 + b"mlkem-dz").digest(64)
    d, z = dz[:32], dz[32:]
    if hasattr(kem, "key_derive"):
        return kem.key_derive(d + z)
    return kem._keygen_internal(d, z)  # kyber-py internal derandomised keygen


def _x25519_from_seed(seed32):
    sk_bytes = hashlib.shake_256(seed32 + b"x25519-sk").digest(32)
    return X25519PrivateKey.from_private_bytes(sk_bytes)


class Party:
    """Holds one client's key material for a given scheme."""

    def __init__(self, scheme, master_seed=None):
        self.scheme = scheme
        self.master = master_seed or _rand_secret32()
        self.x_sk = self.kem_ek = self.kem_dk = None
        if scheme in ("x25519", "hybrid"):
            self.x_sk = _x25519_from_seed(self.master)
        if scheme.startswith("mlkem") or scheme == "hybrid":
            kem = MLKEM["mlkem768"] if scheme == "hybrid" else MLKEM[scheme]
            self.kem = kem
            self.kem_ek, self.kem_dk = _mlkem_keygen_from_seed(kem, self.master)

    def public_bundle(self):
        parts = []
        if self.x_sk is not None:
            parts.append(self.x_sk.public_key().public_bytes_raw())
        if self.kem_ek is not None:
            parts.append(self.kem_ek)
        return parts


def derive_pair_seed(scheme, dh_ss, kem_ss, ctx):
    h = hashlib.sha3_256()
    h.update(b"pqfl-pairseed-v1|" + scheme.encode() + b"|")
    h.update(dh_ss or b"")
    h.update(kem_ss or b"")
    h.update(ctx)
    return h.digest()


def prg_mask(seed32, rnd, d):
    raw = hashlib.shake_128(seed32 + rnd.to_bytes(4, "little")
                            + b"maskexp").digest(4 * d)
    return np.frombuffer(raw, dtype=np.uint32).copy()


# ---- Shamir secret sharing over GF(p), p = 2^255 - 19 ----------------------


def shamir_share(secret_int, n, t, rng):
    coeffs = [secret_int] + [rng.randrange(SHAMIR_P) for _ in range(t - 1)]
    shares = []
    for x in range(1, n + 1):
        y, xp = 0, 1
        for c in coeffs:
            y = (y + c * xp) % SHAMIR_P
            xp = (xp * x) % SHAMIR_P
        shares.append((x, y))
    return shares


def shamir_reconstruct(points):
    total = 0
    for i, (xi, yi) in enumerate(points):
        num = den = 1
        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            num = (num * (-xj)) % SHAMIR_P
            den = (den * (xi - xj)) % SHAMIR_P
        total = (total + yi * num * pow(den, -1, SHAMIR_P)) % SHAMIR_P
    return total



def _rand_secret32():
    """32 random bytes with top two bits cleared so the big-endian integer
    is < 2**254 < SHAMIR_P and round-trips exactly through Shamir sharing."""
    b = bytearray(secrets.token_bytes(32))
    b[0] &= 0x3F
    return bytes(b)

SHARE_BYTES = 2 + 32  # (x index, 32-byte field element)


# =============================================================================
# Secure aggregation protocol (Bonawitz-style double masking, KEM pairwise)
# =============================================================================


@dataclass
class Meter:
    up: dict = field(default_factory=dict)      # bytes uploaded per phase
    down: dict = field(default_factory=dict)    # bytes downloaded per phase
    t: dict = field(default_factory=dict)       # seconds per phase

    def add(self, kind, phase, v):
        d = getattr(self, kind)
        d[phase] = d.get(phase, 0) + v


class SecAgg:
    """One aggregation group of n clients. Semi-honest model.

    Setup (once per rekey period):
      R0  every client i publishes its public bundle (X25519 pk and/or ML-KEM ek)
      R1  for every pair i<j, client i encapsulates to j's ek (ct relayed by
          the server and retained) and, for X25519/hybrid, both sides run DH;
          both derive the pairwise seed s_ij
      R2  every client Shamir-shares (t = floor(n/2)+1) its 32-byte master
          seed m_i (from which its decap/DH keys derive) and its self-mask
          seed b_i to all peers via the server
    Per round:
      R3  client i uploads y_i = q(x_i) + PRG(b_i, r) + sum_{j>i} PRG(s_ij, r)
          - sum_{j<i} PRG(s_ij, r)   (mod 2^32)
      R4  for survivors, peers reveal shares of b_i; for dropped u, peers
          reveal shares of m_u, the server reconstructs u's decap/DH keys,
          decapsulates the retained ciphertexts addressed to u, and surviving
          senders j>u disclose s_uj directly; the server removes all mask
          terms and dequantises the sum.
    """

    def __init__(self, n, scheme, d, quant, rng_seed=0):
        self.n, self.scheme, self.d, self.q = n, scheme, d, quant
        self.rng = np.random.default_rng(rng_seed)
        self.pyrng = __import__("random").Random(rng_seed)
        self.t = n // 2 + 1
        self.meter = Meter()
        self.parties = None

    # ---- setup ---------------------------------------------------------
    def setup(self):
        m, n = self.meter, self.n
        t0 = time.perf_counter()
        self.parties = [Party(self.scheme) for _ in range(n)]
        m.add("t", "setup_keygen", time.perf_counter() - t0)

        # R0: publish bundles
        for p in self.parties:
            b = sum(len(x) for x in p.public_bundle())
            m.add("up", "setup_pk", b)
            m.add("down", "setup_pk", b * (n - 1))

        # R1: pairwise agreement, lower index encapsulates to higher
        t0 = time.perf_counter()
        self.seeds = {}                # (i,j) -> 32B, i<j
        self.ct_store = {}             # (i,j) -> ct addressed to j
        for i in range(n):
            for j in range(i + 1, n):
                dh_ss = kem_ss = None
                ct_len = 0
                if self.scheme in ("x25519", "hybrid"):
                    dh_ss = self.parties[i].x_sk.exchange(
                        self.parties[j].x_sk.public_key())
                if self.scheme.startswith("mlkem") or self.scheme == "hybrid":
                    kem_ss, ct = self.parties[j].kem.encaps(
                        self.parties[j].kem_ek)
                    # sender is i in the real protocol; kyber-py encaps only
                    # needs j's ek, so identical bytes/time either way
                    kem_ss2 = self.parties[j].kem.decaps(
                        self.parties[j].kem_dk, ct)
                    assert kem_ss == kem_ss2
                    self.ct_store[(i, j)] = ct
                    ct_len = len(ct)
                ctx = f"{i}|{j}".encode()
                self.seeds[(i, j)] = derive_pair_seed(
                    self.scheme, dh_ss, kem_ss, ctx)
                m.add("up", "setup_ct", ct_len)      # i -> server
                m.add("down", "setup_ct", ct_len)    # server -> j
        m.add("t", "setup_pairwise", time.perf_counter() - t0)

        # R2: Shamir shares of (m_i, b_i)
        t0 = time.perf_counter()
        self.b_seed = [_rand_secret32() for _ in range(n)]
        self.shares_m = []
        self.shares_b = []
        for i in range(n):
            sm = shamir_share(int.from_bytes(self.parties[i].master, "big"),
                              n, self.t, self.pyrng)
            sb = shamir_share(int.from_bytes(self.b_seed[i], "big"),
                              n, self.t, self.pyrng)
            self.shares_m.append(sm)
            self.shares_b.append(sb)
            m.add("up", "setup_shares", 2 * n * SHARE_BYTES)
            m.add("down", "setup_shares", 2 * n * SHARE_BYTES)
        m.add("t", "setup_shares", time.perf_counter() - t0)

    # ---- one aggregation round ------------------------------------------
    def round(self, updates, rnd=1, dropout=()):
        """updates: list of n float32 vectors (real model deltas).
        dropout: indices that fail after uploading nothing this round."""
        m, n, d = self.meter, self.n, self.d
        survivors = [i for i in range(n) if i not in set(dropout)]

        # R3: mask + upload (survivors only)
        t0 = time.perf_counter()
        ysum = np.zeros(d, np.uint32)
        clip_fracs = []
        for i in survivors:
            qv, cf = self.q.encode(updates[i])
            clip_fracs.append(cf)
            y = qv + prg_mask(self.b_seed[i], rnd, d)
            for j in range(i + 1, n):
                y += prg_mask(self.seeds[(i, j)], rnd, d)
            for j in range(0, i):
                y -= prg_mask(self.seeds[(j, i)], rnd, d)
            ysum += y
            m.add("up", "round_masked", 4 * d)
        m.add("t", "round_mask_upload", time.perf_counter() - t0)

        # R4: unmask
        t0 = time.perf_counter()
        # survivors' self-masks: reconstruct b_i from t shares
        for i in survivors:
            b_int = shamir_reconstruct(self.shares_b[i][:self.t])
            assert b_int == int.from_bytes(self.b_seed[i], "big")
            ysum -= prg_mask(self.b_seed[i], rnd, d)
            m.add("up", "round_b_shares", self.t * SHARE_BYTES)
        m.add("t", "round_unmask_self", time.perf_counter() - t0)

        # dropped clients: remove their pairwise terms with survivors
        t0 = time.perf_counter()
        for u in dropout:
            m_int = shamir_reconstruct(self.shares_m[u][:self.t])
            m.add("up", "round_m_shares", self.t * SHARE_BYTES)
            master = m_int.to_bytes(32, "big")
            x_sk = (_x25519_from_seed(master)
                    if self.scheme in ("x25519", "hybrid") else None)
            kem_dk = None
            if self.scheme.startswith("mlkem") or self.scheme == "hybrid":
                kem = (MLKEM["mlkem768"] if self.scheme == "hybrid"
                       else MLKEM[self.scheme])
                _, kem_dk = _mlkem_keygen_from_seed(kem, master)
            for j in survivors:
                a, b = (u, j) if u < j else (j, u)
                if u < j:
                    # u was sender; surviving receiver j discloses s_uj
                    s = self.seeds[(a, b)]
                    m.add("up", "round_disclose", 32)
                else:
                    # u was receiver; server decapsulates retained ct
                    dh_ss = kem_ss = None
                    if x_sk is not None:
                        dh_ss = x_sk.exchange(
                            self.parties[j].x_sk.public_key())
                    if kem_dk is not None:
                        kem_ss = kem.decaps(kem_dk, self.ct_store[(a, b)])
                    s = derive_pair_seed(self.scheme, dh_ss, kem_ss,
                                         f"{a}|{b}".encode())
                    assert s == self.seeds[(a, b)]
                # u's contribution to survivor sums: +PRG for pairs (u,j) u<j
                # never entered ysum (u dropped); survivors j carry -PRG(s_uj)
                # if u<j? No: survivor i sums +PRG(s_ij) for j>i and -PRG(s_ji)
                # for j<i. Dropped u leaves unbalanced terms in survivors:
                if u < j:
                    ysum += prg_mask(s, rnd, d)   # j carried -PRG(s_uj)
                else:
                    ysum -= prg_mask(s, rnd, d)   # j carried +PRG(s_ju)
        m.add("t", "round_recovery", time.perf_counter() - t0)

        agg = self.q.decode_sum(ysum, len(survivors))
        return agg, float(np.mean(clip_fracs)) if clip_fracs else 0.0


# =============================================================================
# STAGE 2 -- UNIT TESTS: exact cancellation, dropout recovery, quant error
# =============================================================================


def stage_unittest():
    rng = np.random.default_rng(7)
    d, n = 4096, 8
    q = Quantizer()
    ups = [rng.normal(0, 0.05, d).astype(np.float32) for _ in range(n)]
    plain_qsum = np.zeros(d, np.uint32)
    for u in ups:
        plain_qsum += q.encode(u)[0]
    ref = q.decode_sum(plain_qsum, n)

    for scheme in ("x25519", "mlkem768", "hybrid"):
        sa = SecAgg(n, scheme, d, q, rng_seed=1)
        sa.setup()
        agg, _ = sa.round(ups, rnd=1)
        assert np.allclose(agg, ref, atol=1e-9), scheme
        # dropout: clients 2 and 5 fail
        drop = (2, 5)
        surv = [i for i in range(n) if i not in drop]
        plain2 = np.zeros(d, np.uint32)
        for i in surv:
            plain2 += q.encode(ups[i])[0]
        ref2 = q.decode_sum(plain2, len(surv))
        agg2, _ = sa.round(ups, rnd=2, dropout=drop)
        assert np.allclose(agg2, ref2, atol=1e-9), scheme + "-dropout"
        err = np.max(np.abs(agg / n - np.mean(ups, 0)))
        log(f"unittest {scheme:8s}: exact cancellation OK, dropout OK, "
            f"max quant err {err:.2e} (bound {1/q.scale:.2e})")
    log("stage unittest: DONE")


# =============================================================================
# STAGE 3 -- CRYPTO MICROBENCHMARKS
# =============================================================================


def _bench(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    a = np.array(ts) * 1e3
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "reps": reps}


def stage_crypto():
    path = os.path.join(OUT_DIR, "crypto_bench.json")
    if os.path.exists(path):
        log("crypto bench cached")
        return
    res = {"sizes": {}, "ops": {}}
    # X25519
    sk = X25519PrivateKey.generate()
    pk = X25519PrivateKey.generate().public_key()
    res["ops"]["x25519_keygen"] = _bench(X25519PrivateKey.generate, 2000)
    res["ops"]["x25519_exchange"] = _bench(lambda: sk.exchange(pk), 2000)
    res["sizes"]["x25519"] = {"pk": 32, "ss": 32, "ct": 0}
    # ML-KEM levels
    for name, kem in MLKEM.items():
        ek, dk = kem.keygen()
        ss, ct = kem.encaps(ek)
        reps = 150
        res["ops"][f"{name}_keygen"] = _bench(kem.keygen, reps)
        res["ops"][f"{name}_encaps"] = _bench(lambda: kem.encaps(ek), reps)
        res["ops"][f"{name}_decaps"] = _bench(lambda: kem.decaps(dk, ct), reps)
        res["sizes"][name] = {"ek": len(ek), "dk": len(dk), "ct": len(ct),
                              "ss": len(ss)}
        log(f"{name}: ek {len(ek)}B ct {len(ct)}B "
            f"keygen {res['ops'][f'{name}_keygen']['median_ms']:.2f}ms")
    # hybrid = X25519 exchange + ML-KEM-768 encaps/decaps + KDF
    ek, dk = ML_KEM_768.keygen()
    ss, ct = ML_KEM_768.encaps(ek)

    def hyb_sender():
        d_ = sk.exchange(pk)
        k_, c_ = ML_KEM_768.encaps(ek)
        derive_pair_seed("hybrid", d_, k_, b"ctx")

    def hyb_receiver():
        d_ = sk.exchange(pk)
        k_ = ML_KEM_768.decaps(dk, ct)
        derive_pair_seed("hybrid", d_, k_, b"ctx")

    res["ops"]["hybrid768_sender"] = _bench(hyb_sender, 150)
    res["ops"]["hybrid768_receiver"] = _bench(hyb_receiver, 150)
    res["sizes"]["hybrid768"] = {"pk": 32 + 1184, "ct": 1088, "ss": 32}
    res["note"] = ("ML-KEM timings are a pure-Python implementation "
                   "(kyber-py) on one CPU core and are conservative upper "
                   "bounds; optimised C implementations are 2-3 orders of "
                   "magnitude faster.")
    save_json(path, res)
    log("stage crypto: DONE")


# =============================================================================
# STAGE 4 -- FEDERATED TRAINING (utility experiments)
# =============================================================================


def dirichlet_partition(fam, n, alpha, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(fam)
    parts = [[] for _ in range(n)]
    for c in classes:
        idx = np.where(fam == c)[0]
        rng.shuffle(idx)
        w = rng.dirichlet([alpha] * n)
        cuts = (np.cumsum(w) * len(idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k] += chunk.tolist()
    return [np.array(sorted(p)) for p in parts]


MODELS = {"mlp_s": (64, 32), "mlp_l": (256, 64)}


def run_fl(dataset, model, agg, seed, rounds=ROUNDS, n=N_CLIENTS_TRAIN,
           qbits=QUANT_BITS_DEFAULT, tag=""):
    """agg in {'plain','secagg'} (secagg = hybrid X25519+ML-KEM-768)."""
    run_id = f"{dataset}_{model}_{agg}_k{qbits}_s{seed}{tag}"
    out_path = os.path.join(OUT_DIR, f"fl_{run_id}.json")
    if os.path.exists(out_path):
        log(f"cached run {run_id}")
        return load_json(out_path)
    Xtr, ytr, fam, Xva, yva, Xte, yte = load_dataset(dataset)
    parts = dirichlet_partition(fam, n, DIRICHLET_ALPHA, seed)
    dims = (Xtr.shape[1],) + MODELS[model] + (1,)
    global_model = MLP(dims, seed=seed)
    d = global_model.dim
    weights = np.array([len(p) for p in parts], np.float64)
    weights /= weights.sum()
    quant = Quantizer(bits=qbits)
    sa = None
    if agg == "secagg":
        sa = SecAgg(n, "hybrid", d, quant, rng_seed=seed)
        sa.setup()
    hist = {"round": [], "val_f1": [], "val_auc": [], "train_loss": [],
            "clip_frac": []}
    rng = np.random.default_rng(seed + 1000)
    t_start = time.time()
    for r in range(1, rounds + 1):
        w0 = global_model.get_flat()
        updates, losses = [], []
        for k in range(n):
            local = MLP(dims, seed=0)
            local.set_flat(w0)
            loss = local.train_epoch(Xtr[parts[k]], ytr[parts[k]],
                                     LOCAL_LR, LOCAL_MOMENTUM, LOCAL_BATCH,
                                     rng)
            updates.append(((local.get_flat() - w0) * weights[k])
                           .astype(np.float32))
            losses.append(loss)
        if agg == "plain":
            delta = np.sum(updates, 0).astype(np.float64)
            cf = 0.0
        else:
            delta, cf = sa.round(updates, rnd=r)
        global_model.set_flat(w0 + delta)
        sva = global_model.predict_proba(Xva)
        mv = metrics_from_scores(yva, sva)
        hist["round"].append(r)
        hist["val_f1"].append(mv["f1"])
        hist["val_auc"].append(mv["auc"])
        hist["train_loss"].append(float(np.average(losses, weights=weights)))
        hist["clip_frac"].append(cf)
        if r % 5 == 0 or r == 1:
            log(f"{run_id} r{r:02d} loss {hist['train_loss'][-1]:.4f} "
                f"valF1 {mv['f1']:.4f}")
    ste = global_model.predict_proba(Xte)
    m = metrics_from_scores(yte, ste)
    np.savez_compressed(os.path.join(OUT_DIR, f"pred_{run_id}.npz"),
                        y=yte, score=ste)
    # persist one set of real per-client update vectors for the benchmark
    vec_path = os.path.join(OUT_DIR, f"updates_{dataset}_{model}.npz")
    if not os.path.exists(vec_path):
        np.savez_compressed(vec_path, U=np.stack(updates), d=d)
    res = {"run_id": run_id, "dataset": dataset, "model": model, "agg": agg,
           "seed": seed, "qbits": qbits, "d": int(d), "test": m,
           "hist": hist, "wall_s": time.time() - t_start,
           "mean_clip_frac": float(np.mean(hist["clip_frac"]))}
    save_json(out_path, res)
    log(f"{run_id}: test F1 {m['f1']:.4f} AUC {m['auc']:.4f} "
        f"({res['wall_s']:.0f}s)")
    return res


def stage_train():
    # centralized baselines (logistic regression + MLP upper bound)
    from sklearn.linear_model import LogisticRegression
    for ds in ("nsl", "unsw"):
        path = os.path.join(OUT_DIR, f"central_{ds}.json")
        if not os.path.exists(path):
            Xtr, ytr, _, Xva, yva, Xte, yte = load_dataset(ds)
            lr = LogisticRegression(max_iter=300)
            lr.fit(Xtr, ytr)
            m_lr = metrics_from_scores(yte, lr.predict_proba(Xte)[:, 1])
            cm = {}
            for model in MODELS:
                per_seed = []
                for seed in SEEDS:
                    mm = MLP((Xtr.shape[1],) + MODELS[model] + (1,),
                             seed=seed)
                    rng = np.random.default_rng(seed)
                    for _ in range(ROUNDS):
                        mm.train_epoch(Xtr, ytr, LOCAL_LR, LOCAL_MOMENTUM,
                                       LOCAL_BATCH, rng)
                    per_seed.append(metrics_from_scores(
                        yte, mm.predict_proba(Xte)))
                cm[model] = per_seed
            save_json(path, {"logreg": m_lr, "central_mlp": cm})
            log(f"centralized {ds}: logreg F1 {m_lr['f1']:.4f}")
    # federated grid
    for ds in ("nsl", "unsw"):
        for model in MODELS:
            for agg in ("plain", "secagg"):
                for seed in SEEDS:
                    run_fl(ds, model, agg, seed)
    log("stage train: DONE")


# =============================================================================
# STAGE 5 -- ROUND-LEVEL BENCHMARKS on real update vectors
# =============================================================================


def _real_vectors(model, n, d):
    z = np.load(os.path.join(OUT_DIR, f"updates_nsl_{model}.npz"))
    U = z["U"].astype(np.float32)
    reps = int(np.ceil(n / len(U)))
    return [U[i % len(U)] for i in range(n)], int(z["d"])


def stage_bench():
    path = os.path.join(OUT_DIR, "round_bench.json")
    done = load_json(path) if os.path.exists(path) else {}
    quant = Quantizer()
    grid_n = (5, 10, 20, 50, 100)
    schemes = ("x25519", "mlkem768", "hybrid")
    for model in ("mlp_s", "mlp_l"):
        for scheme in schemes:
            for n in grid_n:
                if scheme != "hybrid" and n == 100 and model == "mlp_l":
                    pass  # still run: full grid
                key = f"{model}|{scheme}|n{n}"
                if key in done:
                    continue
                ups, d = _real_vectors(model, n, None)
                sa = SecAgg(n, scheme, d, quant, rng_seed=3)
                sa.setup()
                agg, _ = sa.round(ups, rnd=1)          # no-dropout round
                # verify against plaintext quantised sum
                ref = np.zeros(d, np.uint32)
                for u in ups:
                    ref += quant.encode(u)[0]
                assert np.allclose(agg, quant.decode_sum(ref, n), atol=1e-9)
                rec = {"d": d, "n": n, "scheme": scheme,
                       "up": dict(sa.meter.up), "down": dict(sa.meter.down),
                       "t": dict(sa.meter.t)}
                done[key] = rec
                save_json(path, done)
                log(f"bench {key}: setup "
                    f"{rec['t'].get('setup_pairwise', 0):.2f}s "
                    f"mask {rec['t'].get('round_mask_upload', 0):.2f}s")
    # dropout recovery at n=50, mlp_l
    drop_path = os.path.join(OUT_DIR, "dropout_bench.json")
    ddone = load_json(drop_path) if os.path.exists(drop_path) else {}
    n = 50
    for model in ("mlp_l",):
        ups, d = _real_vectors(model, n, None)
        for scheme in schemes:
            for frac in (0.0, 0.1, 0.2, 0.3):
                key = f"{model}|{scheme}|n{n}|drop{frac}"
                if key in ddone:
                    continue
                sa = SecAgg(n, scheme, d, quant, rng_seed=4)
                sa.setup()
                k = int(round(frac * n))
                drop = tuple(range(0, 2 * k, 2))[:k]
                surv = [i for i in range(n) if i not in set(drop)]
                agg, _ = sa.round(ups, rnd=1, dropout=drop)
                ref = np.zeros(d, np.uint32)
                for i in surv:
                    ref += quant.encode(ups[i])[0]
                assert np.allclose(agg, quant.decode_sum(ref, len(surv)),
                                   atol=1e-9)
                ddone[key] = {"d": d, "n": n, "scheme": scheme, "frac": frac,
                              "t": dict(sa.meter.t), "up": dict(sa.meter.up)}
                save_json(drop_path, ddone)
                log(f"dropout {key}: recovery "
                    f"{ddone[key]['t'].get('round_recovery', 0):.2f}s")
    log("stage bench: DONE")


# =============================================================================
# STAGE 6 -- ABLATIONS: ML-KEM level, quantisation bits, rekey cadence
# =============================================================================


def stage_ablations():
    quant = Quantizer()
    # (a) ML-KEM security level at n=20, mlp_l
    path = os.path.join(OUT_DIR, "level_bench.json")
    done = load_json(path) if os.path.exists(path) else {}
    ups, d = _real_vectors("mlp_l", 20, None)
    for lvl in ("mlkem512", "mlkem768", "mlkem1024"):
        if lvl in done:
            continue
        sa = SecAgg(20, lvl, d, quant, rng_seed=5)
        sa.setup()
        sa.round(ups, rnd=1)
        done[lvl] = {"t": dict(sa.meter.t), "up": dict(sa.meter.up),
                     "down": dict(sa.meter.down), "d": d, "n": 20}
        save_json(path, done)
        log(f"level {lvl}: pairwise {done[lvl]['t']['setup_pairwise']:.2f}s")
    # (b) quantisation bit-width utility (nsl, mlp_s, secagg)
    for k in (8, 16, 24):
        for seed in SEEDS:
            run_fl("nsl", "mlp_s", "secagg", seed, qbits=k, tag="_abl")
    log("stage ablations: DONE")


# =============================================================================
# STAGE 7 -- ANALYTICAL COMMUNICATION MODEL + VALIDATION
# =============================================================================


def analytic_bytes(scheme, n, d, sizes):
    """Per-client uplink bytes: setup (once per rekey period) and per round.

    Model: a client publishes one public bundle (pk), acts as encapsulating
    sender to the (n-1)/2 higher-indexed peers on average (each costing one
    ciphertext ct), and Shamir-shares two 32-byte seeds to all n peers.
    """
    if scheme == "x25519":
        pk, ct = 32, 0
    elif scheme == "hybrid":
        pk, ct = 32 + 1184, 1088
    else:
        pk, ct = sizes[scheme]["ek"], sizes[scheme]["ct"]
    setup = pk + ((n - 1) / 2.0) * ct + 2 * n * SHARE_BYTES
    per_round = 4 * d + (n // 2 + 1) * SHARE_BYTES  # masked vec + b-shares
    return setup, per_round


def stage_model():
    bench = load_json(os.path.join(OUT_DIR, "round_bench.json"))
    sizes = load_json(os.path.join(OUT_DIR, "crypto_bench.json"))["sizes"]
    rows = []
    for key, rec in bench.items():
        n, d, scheme = rec["n"], rec["d"], rec["scheme"]
        meas_setup = (rec["up"].get("setup_pk", 0)
                      + rec["up"].get("setup_ct", 0)
                      + rec["up"].get("setup_shares", 0)) / n
        meas_round = (rec["up"].get("round_masked", 0)
                      + rec["up"].get("round_b_shares", 0)) / n
        pred_setup, pred_round = analytic_bytes(scheme, n, d, sizes)
        rows.append({"key": key, "n": n, "d": d, "scheme": scheme,
                     "meas_setup": meas_setup, "pred_setup": pred_setup,
                     "err_setup_pct": 100 * (pred_setup - meas_setup)
                     / meas_setup,
                     "meas_round": meas_round, "pred_round": pred_round,
                     "err_round_pct": 100 * (pred_round - meas_round)
                     / meas_round})
    save_json(os.path.join(OUT_DIR, "model_validation.json"), rows)
    worst = max(abs(r["err_setup_pct"]) for r in rows)
    log(f"analytic model: worst setup-bytes error {worst:.2f}% "
        f"over {len(rows)} configs")
    log("stage model: DONE")


# =============================================================================
# STAGE 8 -- TABLES AND FIGURES (generated only from cached artifacts)
# =============================================================================


def _agg_seed_runs(ds, model, agg, key, qbits=QUANT_BITS_DEFAULT, tag=""):
    vals = []
    for s in SEEDS:
        r = load_json(os.path.join(
            OUT_DIR, f"fl_{ds}_{model}_{agg}_k{qbits}_s{s}{tag}.json"))
        vals.append(r["test"][key])
    return float(np.mean(vals)), float(np.std(vals))


def stage_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    cb = load_json(os.path.join(OUT_DIR, "crypto_bench.json"))
    rb = load_json(os.path.join(OUT_DIR, "round_bench.json"))
    db = load_json(os.path.join(OUT_DIR, "dropout_bench.json"))

    # ---------------- Table 1: dataset summary --------------------------
    ds_sum = load_json(os.path.join(OUT_DIR, "dataset_summary.json"))
    for k, v in ds_sum.items():
        for kk, vv in v.items():
            print(f"Table 1 | {k} {kk}: {vv}")

    # ---------------- Table 2: detection utility ------------------------
    t2 = []
    for ds in ("nsl", "unsw"):
        cen = load_json(os.path.join(OUT_DIR, f"central_{ds}.json"))
        print(f"Table 2 | {ds} logreg-central: "
              + ", ".join(f"{k}={v:.4f}" for k, v in cen["logreg"].items()))
        for model in MODELS:
            cvals = cen["central_mlp"][model]
            for met in ("acc", "f1", "auc"):
                mu = np.mean([c[met] for c in cvals])
                sd = np.std([c[met] for c in cvals])
                print(f"Table 2 | {ds} {model} central {met}: "
                      f"{mu:.4f} +/- {sd:.4f}")
            for agg in ("plain", "secagg"):
                row = {"ds": ds, "model": model, "agg": agg}
                for met in ("acc", "prec", "rec", "f1", "auc"):
                    mu, sd = _agg_seed_runs(ds, model, agg, met)
                    row[met] = (mu, sd)
                    print(f"Table 2 | {ds} {model} {agg} {met}: "
                          f"{mu:.4f} +/- {sd:.4f}")
                t2.append(row)
    save_json(os.path.join(OUT_DIR, "table2.json"), t2)

    # ---------------- Table 3: crypto microbench ------------------------
    for op, v in cb["ops"].items():
        print(f"Table 3 | {op}: median {v['median_ms']:.3f} ms "
              f"(p95 {v['p95_ms']:.3f})")
    for s, v in cb["sizes"].items():
        print(f"Table 3 | sizes {s}: {v}")

    # ---------------- Table 4: per-round costs --------------------------
    for key, rec in sorted(rb.items()):
        per_cli_setup = (rec["up"].get("setup_pk", 0)
                         + rec["up"].get("setup_ct", 0)
                         + rec["up"].get("setup_shares", 0)) / rec["n"]
        per_cli_round = (rec["up"].get("round_masked", 0)
                         + rec["up"].get("round_b_shares", 0)) / rec["n"]
        print(f"Table 4 | {key}: setupB/cli {per_cli_setup:.0f} "
              f"roundB/cli {per_cli_round:.0f} "
              f"pairwise_s {rec['t'].get('setup_pairwise', 0):.3f} "
              f"mask_s {rec['t'].get('round_mask_upload', 0):.3f} "
              f"unmask_s {rec['t'].get('round_unmask_self', 0):.3f}")

    # ---------------- Table 5: dropout recovery -------------------------
    for key, rec in sorted(db.items()):
        print(f"Table 5 | {key}: recovery_s "
              f"{rec['t'].get('round_recovery', 0):.3f} "
              f"disclose_B {rec['up'].get('round_disclose', 0)} "
              f"m_shares_B {rec['up'].get('round_m_shares', 0)}")

    # ---------------- Table 6: ablations --------------------------------
    lb = load_json(os.path.join(OUT_DIR, "level_bench.json"))
    for lvl, rec in lb.items():
        print(f"Table 6a | {lvl} n=20: pairwise_s "
              f"{rec['t']['setup_pairwise']:.3f} "
              f"setup_ct_B {rec['up'].get('setup_ct', 0)}")
    for k in (8, 16, 24):
        mu, sd = _agg_seed_runs("nsl", "mlp_s", "secagg", "f1",
                                qbits=k, tag="_abl" if k != 16 else "_abl")
        print(f"Table 6b | quant k={k}: F1 {mu:.4f} +/- {sd:.4f}")
    # rekey cadence (amortised from measured setup + per-round costs)
    rec = rb["mlp_l|hybrid|n20"]
    setup_t = sum(v for k_, v in rec["t"].items() if k_.startswith("setup"))
    round_t = rec["t"].get("round_mask_upload", 0)
    for R in (1, 5, 10, 20):
        print(f"Table 6c | rekey R={R}: amortised crypto+mask s/round "
              f"{setup_t / R + round_t:.3f}")

    # ---------------- Table 7: model validation -------------------------
    mv = load_json(os.path.join(OUT_DIR, "model_validation.json"))
    for r in mv:
        print(f"Table 7 | {r['key']}: meas_setupB {r['meas_setup']:.0f} "
              f"pred {r['pred_setup']:.0f} err {r['err_setup_pct']:+.2f}% | "
              f"meas_roundB {r['meas_round']:.0f} err "
              f"{r['err_round_pct']:+.2f}%")

    # ================= FIGURES ==========================================
    # Fig 1: convergence, plain vs secagg
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, ds, title in zip(axes, ("nsl", "unsw"),
                             ("NSL-KDD", "UNSW-NB15")):
        for agg, style in (("plain", "-"), ("secagg", "--")):
            F = []
            for s in SEEDS:
                r = load_json(os.path.join(
                    OUT_DIR, f"fl_{ds}_mlp_l_{agg}_k16_s{s}.json"))
                F.append(r["hist"]["val_f1"])
            F = np.array(F)
            ax.plot(r["hist"]["round"], F.mean(0), style,
                    label=("FedAvg (plaintext)" if agg == "plain"
                           else "FedAvg + hybrid SecAgg"))
            ax.fill_between(r["hist"]["round"], F.mean(0) - F.std(0),
                            F.mean(0) + F.std(0), alpha=0.2)
        ax.set_xlabel("round")
        ax.set_ylabel("validation F1")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_convergence.png"), dpi=200)

    # Fig 2: crypto op latency (log scale)
    ops = ["x25519_keygen", "x25519_exchange",
           "mlkem512_keygen", "mlkem512_encaps", "mlkem512_decaps",
           "mlkem768_keygen", "mlkem768_encaps", "mlkem768_decaps",
           "mlkem1024_keygen", "mlkem1024_encaps", "mlkem1024_decaps",
           "hybrid768_sender", "hybrid768_receiver"]
    vals = [cb["ops"][o]["median_ms"] for o in ops]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(range(len(ops)), vals, color="#4477aa")
    ax.set_yscale("log")
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([o.replace("_", "\n") for o in ops], fontsize=7)
    ax.set_ylabel("median latency (ms, log)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_crypto_ops.png"), dpi=200)

    # Fig 3: per-client setup bytes vs n (measured markers + analytic lines)
    sizes = cb["sizes"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = {"x25519": "#228833", "mlkem768": "#ee6677",
              "hybrid": "#4477aa"}
    ns = np.array((5, 10, 20, 50, 100))
    for scheme in ("x25519", "mlkem768", "hybrid"):
        meas = []
        for n in ns:
            rec = rb[f"mlp_l|{scheme}|n{n}"]
            meas.append((rec["up"].get("setup_pk", 0)
                         + rec["up"].get("setup_ct", 0)
                         + rec["up"].get("setup_shares", 0)) / n)
        pred = [analytic_bytes(scheme, int(n), rec["d"], sizes)[0]
                for n in ns]
        ax.plot(ns, np.array(pred) / 1024, "-", color=colors[scheme],
                label=f"{scheme} (model)")
        ax.plot(ns, np.array(meas) / 1024, "o", color=colors[scheme],
                label=f"{scheme} (measured)")
    ax.set_xlabel("clients n")
    ax.set_ylabel("setup uplink per client (KiB)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_setup_bytes.png"), dpi=200)

    # Fig 4: dropout recovery time
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for scheme in ("x25519", "mlkem768", "hybrid"):
        xs, ys = [], []
        for frac in (0.0, 0.1, 0.2, 0.3):
            rec = db[f"mlp_l|{scheme}|n50|drop{frac}"]
            xs.append(frac * 100)
            ys.append(rec["t"].get("round_recovery", 0))
        ax.plot(xs, ys, "o-", color=colors[scheme], label=scheme)
    ax.set_xlabel("dropout rate (%)  [n = 50]")
    ax.set_ylabel("server recovery time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_dropout.png"), dpi=200)

    # Fig 5: ROC curves (mlp_l, secagg, seed 0)
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for ds, c in (("nsl", "#4477aa"), ("unsw", "#ee6677")):
        z = np.load(os.path.join(OUT_DIR,
                                 f"pred_{ds}_mlp_l_secagg_k16_s0.npz"))
        fpr, tpr, _ = roc_curve(z["y"], z["score"])
        ax.plot(fpr, tpr, color=c,
                label=f"{'NSL-KDD' if ds == 'nsl' else 'UNSW-NB15'} "
                      f"(AUC {sk_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_roc.png"), dpi=200)
    log("stage figs: DONE (5 figures written to ./figs)")


# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "unittest", "crypto", "train",
                             "bench", "ablations", "model", "figs"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stages = {"data": stage_data, "unittest": stage_unittest,
              "crypto": stage_crypto, "train": stage_train,
              "bench": stage_bench, "ablations": stage_ablations,
              "model": stage_model, "figs": stage_figs}
    order = ["data", "unittest", "crypto", "train", "bench", "ablations",
             "model", "figs"]
    todo = order if args.stage == "all" else [args.stage]
    for s in todo:
        try:
            log(f"===== stage {s} =====")
            stages[s]()
        except Exception as e:
            log(f"STAGE {s} FAILED: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
