"""Eksperimen peningkatan F1 classifier EN pada gold standard konsensus.

Protokol (agar skor test tidak bias):
- Split train/val/test sama persis dengan notebook §7 (SEED 42, stratifikasi konstruk paling langka).
- Data dev = train + val. Semua pilihan (fitur, model, hiperparameter, jumlah epoch, threshold) dibuat dengan
  validasi silang 5 lipatan pada dev. Threshold dinilai secara cross-fitted: lipatan k memakai threshold yang
  di-tuning pada prediksi out-of-fold lipatan lain.
- Test set hanya dipakai sekali, oleh perintah `final`.

    .venv/bin/python scripts/eksperimen_f1.py linear                # TF-IDF dan embedding beku + regresi logistik
    .venv/bin/python scripts/eksperimen_f1.py finetune base pw3 ... # fine-tuning IndoBERT per konfigurasi
    .venv/bin/python scripts/eksperimen_f1.py ringkas               # tabel semua kandidat (CV dev)
    .venv/bin/python scripts/eksperimen_f1.py final <kandidat>      # latih ulang pada dev, evaluasi sekali di test
    .venv/bin/python scripts/eksperimen_f1.py kurva                 # kurva belajar (CV dev, tanpa test)
    .venv/bin/python scripts/eksperimen_f1.py gambar_kurva          # grafik kurva belajar
    .venv/bin/python scripts/eksperimen_f1.py gabung                # skenario EN05+EN06 digabung (CV, lalu test sekali)
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch  # sebelum sklearn/scipy: urutan sebaliknya membuat proses segfault saat inferensi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eksperimen_f1"
EN_COLS = [f"EN0{i}" for i in range(1, 9)]
SEED = 42
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

EMBEDDERS = {  # nama -> (model HF, prefix teks)
    "indobert": ("indobenchmark/indobert-base-p2", ""),
    "e5base": ("intfloat/multilingual-e5-base", "query: "),
}
BASE = {"model": "indobenchmark/indobert-base-p2", "pooling": "cls", "lr": 2e-5, "pw_cap": 10.0, "epochs": 10,
        "batch_size": 16, "max_length": 128, "warmup": 0.1}
CONFIGS = {
    "base": BASE,                                             # setelan notebook, 10 epoch
    "pw3": BASE | {"pw_cap": 3.0},
    "mean_pw3": BASE | {"pooling": "mean", "pw_cap": 3.0},
    "lr3_pw3": BASE | {"lr": 3e-5, "pw_cap": 3.0},
    "dapt": BASE | {"model": str(OUT / "dapt")},              # IndoBERT setelah domain-adaptive pretraining
    "dapt_focal": BASE | {"model": str(OUT / "dapt"), "focal_gamma": 2.0},  # + focal loss (Lin dkk., 2017)
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- data dan split ----------
def load_split():
    gold = json.loads((ROOT / "annotation" / "gold_standard.json").read_text(encoding="utf-8"))
    gold_pos = {c: sum(r[c] for r in gold) for c in EN_COLS}
    keys = []
    for r in gold:
        pos = [c for c in EN_COLS if r[c] == 1]
        keys.append(min(pos, key=gold_pos.get) if pos else "none")
    counts = Counter(keys)
    keys = [k if counts[k] >= 10 else "rare" for k in keys]
    if Counter(keys)["rare"] < 10:
        keys = ["none" if k == "rare" else k for k in keys]
    idx = np.arange(len(gold))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=keys)
    try:
        va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=[keys[i] for i in tmp])
    except ValueError:
        va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    dev = np.concatenate([tr, va])
    texts = [r["text"] for r in gold]
    y = np.array([[r[c] for c in EN_COLS] for r in gold], dtype=int)
    folds = [f for _, f in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, [keys[i] for i in dev])]
    return texts, y, dev, te, folds


# ---------- metrik ----------
def tune(y, p):
    th = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        scores = [f1_score(y[:, j], p[:, j] >= t, zero_division=0) for t in GRID]
        if max(scores) > 0:
            th[j] = GRID[int(np.argmax(scores))]
    return th


def crossfit_f1(y, p, folds):
    """F1 per konstruk dengan threshold cross-fitted (lipatan k memakai threshold dari lipatan lain)."""
    pred = np.zeros_like(y)
    for f in folds:
        rest = np.setdiff1d(np.arange(len(y)), f)
        pred[f] = p[f] >= tune(y[rest], p[rest])
    return f1_score(y, pred, average=None, zero_division=0)


# ---------- model linear ----------
def lr_fit_predict(Xtr, ytr, Xte, C):
    out = np.zeros((Xte.shape[0], ytr.shape[1]))
    for j in range(ytr.shape[1]):
        if len(np.unique(ytr[:, j])) < 2:
            continue  # tidak ada contoh positif di data latih (subset kecil pada kurva belajar): prediksi 0
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear")
        clf.fit(Xtr, ytr[:, j])
        out[:, j] = clf.predict_proba(Xte)[:, 1]
    return out


def tfidf_features(train_texts, other_texts):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    Xo = hstack([word.transform(other_texts), char.transform(other_texts)]).tocsr()
    return Xtr, Xo


def embeddings(name, texts):
    path = OUT / f"emb_{name}.npy"
    if path.exists():
        return np.load(path)
    from transformers import AutoModel, AutoTokenizer
    model_name, prefix = EMBEDDERS[name]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok([prefix + t for t in texts[i:i + 32]], max_length=128, truncation=True, padding=True,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    arr = np.concatenate(vecs)
    np.save(path, arr)
    return arr


def oof_linear(feature, texts, y, dev, folds, C):
    oof = np.zeros((len(dev), y.shape[1]))
    dev_texts = [texts[i] for i in dev]
    X = None if feature == "tfidf" else embeddings(feature, texts)[dev]
    for f in folds:
        rest = np.setdiff1d(np.arange(len(dev)), f)
        if feature == "tfidf":
            Xtr, Xte = tfidf_features([dev_texts[i] for i in rest], [dev_texts[i] for i in f])
        else:
            Xtr, Xte = X[rest], X[f]
        oof[f] = lr_fit_predict(Xtr, y[dev][rest], Xte, C)
    return oof


def cmd_linear(texts, y, dev, te, folds):
    ydev = y[dev]
    for feature in ["tfidf", *EMBEDDERS]:
        best = None
        for C in C_GRID:
            oof = oof_linear(feature, texts, y, dev, folds, C)
            f1 = crossfit_f1(ydev, oof, folds)
            log(f"{feature:<9} C={C:<5} CV macro-F1 {f1.mean():.4f}  " + " ".join(f"{v:.2f}" for v in f1))
            if best is None or f1.mean() > best[0]:
                best = (f1.mean(), C, oof)
        np.save(OUT / f"oof_lin_{feature}.npy", best[2])
        (OUT / f"lin_{feature}.json").write_text(json.dumps({"C": best[1], "cv_macro_f1": round(best[0], 4)}))


# ---------- fine-tuning ----------
def build_model(cfg, n_labels):
    from transformers import AutoModel

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(cfg["model"])
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_labels)

        def forward(self, input_ids, attention_mask):
            h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if cfg["pooling"] == "mean":
                m = attention_mask.unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            else:
                v = h[:, 0, :]
            return self.classifier(self.dropout(v))

    return Net()


def train_eval(cfg, tr_texts, ytr, ev_texts, epochs, seed=SEED):
    """Latih pada (tr_texts, ytr); kembalikan probabilitas ev_texts setelah setiap epoch, bentuk (epochs, n, 8)."""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = build_model(cfg, ytr.shape[1]).to(dev)
    pos = ytr.sum(0); neg = len(ytr) - pos
    pw = np.clip(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), 1.0, cfg["pw_cap"])
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float, device=dev), reduction="none")
    gamma = cfg.get("focal_gamma", 0.0)

    def loss_fn(logits, target):
        loss = bce(logits, target)
        if gamma:  # focal: bobot (1 - p_t)^gamma mengecilkan sumbangan contoh yang sudah mudah ditebak
            p = torch.sigmoid(logits)
            p_t = p * target + (1 - p) * (1 - target)
            loss = (1 - p_t) ** gamma * loss
        return loss.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    steps_per_epoch = int(np.ceil(len(tr_texts) / cfg["batch_size"]))
    sched = get_linear_schedule_with_warmup(opt, int(steps_per_epoch * epochs * cfg["warmup"]), steps_per_epoch * epochs)
    rng = np.random.default_rng(seed)

    def enc(batch):
        e = tok(batch, max_length=cfg["max_length"], truncation=True, padding=True, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    order_ev = np.argsort([len(t) for t in ev_texts])
    history = np.zeros((epochs, len(ev_texts), ytr.shape[1]), dtype=np.float32)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_texts))
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            ids, mask = enc([tr_texts[k] for k in b])
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), torch.tensor(ytr[b], dtype=torch.float, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            for i in range(0, len(order_ev), 64):
                b = order_ev[i:i + 64]
                ids, mask = enc([ev_texts[k] for k in b])
                history[ep, b] = torch.sigmoid(model(ids, mask)).cpu().numpy()
    return history, model, tok


def cmd_finetune(names, texts, y, dev, te, folds):
    ydev = y[dev]
    dev_texts = [texts[i] for i in dev]
    for name in names:
        cfg = CONFIGS[name]
        path = OUT / f"oof_ft_{name}.npy"
        if path.exists():
            log(f"{name}: sudah ada, dilewati")
            continue
        t0 = time.time()
        oof = np.zeros((cfg["epochs"], len(dev), y.shape[1]), dtype=np.float32)
        for k, f in enumerate(folds, 1):
            rest = np.setdiff1d(np.arange(len(dev)), f)
            hist, model, _ = train_eval(cfg, [dev_texts[i] for i in rest], ydev[rest], [dev_texts[i] for i in f],
                                        cfg["epochs"])
            oof[:, f] = hist
            del model
            torch.cuda.empty_cache()
            log(f"{name}: lipatan {k}/5 selesai ({(time.time() - t0) / 60:.1f} menit)")
        np.save(path, oof)
        per_epoch = [crossfit_f1(ydev, oof[e], folds).mean() for e in range(cfg["epochs"])]
        log(f"{name}: CV macro-F1 per epoch " + " ".join(f"{v:.3f}" for v in per_epoch))


def cmd_dapt(texts, te, epochs=5):
    """Domain-adaptive pretraining: MLM pada kalimat korpus kurikulum (tanpa label), kalimat test dikeluarkan."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, \
        get_linear_schedule_with_warmup
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    corpus = [r["text"] for name in ("master_corpus.json", "appended_docs.json")
              for r in json.loads((ROOT / "corpus" / name).read_text(encoding="utf-8"))]
    held_out = {texts[i] for i in te}
    corpus = [t for t in dict.fromkeys(corpus) if t not in held_out]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE["model"])
    model = AutoModelForMaskedLM.from_pretrained(BASE["model"]).to(dev)
    collate = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    bs = 16
    steps = int(np.ceil(len(corpus) / bs)) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = np.random.default_rng(SEED)
    log(f"DAPT: {len(corpus)} kalimat (test dikeluarkan: {len(held_out)}), {epochs} epoch")
    for ep in range(epochs):
        model.train()
        perm, total = rng.permutation(len(corpus)), 0.0
        for i in range(0, len(perm), bs):
            enc = tok([corpus[k] for k in perm[i:i + bs]], max_length=128, truncation=True)
            batch = collate([{k: enc[k][j] for k in enc} for j in range(len(enc["input_ids"]))])
            batch = {k: v.to(dev) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item()
        log(f"DAPT epoch {ep + 1}: MLM loss {total / np.ceil(len(corpus) / bs):.4f}")
    model.save_pretrained(OUT / "dapt")
    tok.save_pretrained(OUT / "dapt")


FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def subsets(n_rest, seed):
    """Subset bersarang dari bagian latih satu lipatan: 25% ⊂ 50% ⊂ 75% ⊂ 100%."""
    perm = np.random.default_rng(seed).permutation(n_rest)
    return {f: np.sort(perm[:int(round(f * n_rest))]) for f in FRACTIONS}


def cmd_kurva(texts, y, dev, te, folds):
    """F1 CV per konstruk saat data latih dikurangi. Test tidak dipakai."""
    ydev, dev_texts = y[dev], [texts[i] for i in dev]
    out_path = OUT / "kurva_belajar.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    # TF-IDF: 5 pengulangan subset
    if "tfidf" not in res:
        C = json.loads((OUT / "lin_tfidf.json").read_text())["C"]
        res["tfidf"] = {}
        for f in FRACTIONS:
            runs = []
            for rep in range(5):
                oof = np.zeros(ydev.shape)
                for k, fold in enumerate(folds):
                    rest = np.setdiff1d(np.arange(len(dev)), fold)
                    sub = rest[subsets(len(rest), SEED + 100 * rep + k)[f]]
                    Xtr, Xte = tfidf_features([dev_texts[i] for i in sub], [dev_texts[i] for i in fold])
                    oof[fold] = lr_fit_predict(Xtr, ydev[sub], Xte, C)
                runs.append(crossfit_f1(ydev, oof, folds).tolist())
            res["tfidf"][str(f)] = runs
            log(f"tfidf {f:.2f}: macro {np.mean(runs, axis=0).mean():.3f}")
        out_path.write_text(json.dumps(res))

    # IndoBERT + DAPT: satu pengulangan; 100% memakai OOF yang sudah ada
    cfg = CONFIGS["dapt"]
    res.setdefault("dapt", {})
    for f in FRACTIONS:
        if str(f) in res["dapt"]:
            continue
        if f == 1.0:
            hist = np.load(OUT / "oof_ft_dapt.npy")
        else:
            hist = np.zeros((cfg["epochs"], len(dev), y.shape[1]), dtype=np.float32)
            for k, fold in enumerate(folds):
                rest = np.setdiff1d(np.arange(len(dev)), fold)
                sub = rest[subsets(len(rest), SEED + k)[f]]
                h, model, _ = train_eval(cfg, [dev_texts[i] for i in sub], ydev[sub],
                                         [dev_texts[i] for i in fold], cfg["epochs"])
                hist[:, fold] = h
                del model
                torch.cuda.empty_cache()
        per_epoch = [crossfit_f1(ydev, hist[e], folds) for e in range(cfg["epochs"])]
        best = int(np.argmax([x.mean() for x in per_epoch]))
        res["dapt"][str(f)] = {"epoch": best + 1, "f1": per_epoch[best].tolist()}
        log(f"dapt {f:.2f}: macro {per_epoch[best].mean():.3f} (epoch {best + 1})")
        out_path.write_text(json.dumps(res))
    res["n_train"] = {str(f): int(round(f * len(dev) * 0.8)) for f in FRACTIONS}
    res["train_pos"] = {str(f): (ydev.sum(0) * 0.8 * f).round(1).tolist() for f in FRACTIONS}
    out_path.write_text(json.dumps(res))


EN_NAMES = {"EN01": "Attention", "EN02": "Learning & Memory", "EN03": "Executive Function", "EN04": "Metacognition",
            "EN05": "Emotion/Affect", "EN06": "Motivation/Agency", "EN07": "Social Cognition", "EN08": "Multimodal"}


def cmd_gambar_kurva():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.loads((OUT / "kurva_belajar.json").read_text())
    fr = [str(f) for f in FRACTIONS]
    x = [res["n_train"][f] for f in fr]
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#1f2328", "#59636e", "#e6e8eb"
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5), sharex=True, sharey=True)
    panels = [("macro", "Macro-F1 (8 konstruk)")] + [(c, f"{c} {EN_NAMES[c]}") for c in EN_COLS]
    for ax, (key, title) in zip(axes.flat, panels):
        def col(vals):
            v = np.array(vals)
            return v.mean(-1) if key == "macro" else v[..., EN_COLS.index(key)]
        tf = np.array([col(res["tfidf"][f]) for f in fr])          # (fraksi, pengulangan)
        bert = np.array([col(res["dapt"][f]["f1"]) for f in fr])
        ax.fill_between(x, tf.min(1), tf.max(1), color=orange, alpha=0.15, linewidth=0)
        ax.plot(x, tf.mean(1), color=orange, linewidth=2, marker="o", markersize=5, label="TF-IDF + regresi logistik (rata-rata 5 ulangan, pita = min–maks)")
        ax.plot(x, bert, color=blue, linewidth=2, marker="o", markersize=5, label="IndoBERT + DAPT")
        pos = "" if key == "macro" else f"\n{res['train_pos']['1.0'][EN_COLS.index(key)]:.0f} positif latih pada 100%"
        ax.set_title(title + pos, fontsize=10, color=ink, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Kalimat latih per lipatan", color=muted)
        ax.set_xticks(x)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 (CV 5 lipatan, dev)", color=muted)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Kurva belajar: F1 saat data latih dikurangi (test set tidak dipakai)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "kurva_belajar.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def cmd_gabung(texts, y, dev, te, folds):
    """Skenario: EN05 dan EN06 digabung menjadi satu konstruk Afektif-Motivasional (1 jika salah satunya 1).
    Model IndoBERT + DAPT dilatih ulang untuk 7 label; epoch dipilih dengan CV, test dibuka sekali."""
    cols = ["EN01", "EN02", "EN03", "EN04", "EN05+EN06", "EN07", "EN08"]
    y7 = np.column_stack([y[:, :4], y[:, 4] | y[:, 5], y[:, 6:]])
    ydev, dev_texts, te_texts = y7[dev], [texts[i] for i in dev], [texts[i] for i in te]
    cfg = CONFIGS["dapt"]
    path = OUT / "oof_gabung_dapt.npy"
    if path.exists():
        oof = np.load(path)
    else:
        oof = np.zeros((cfg["epochs"], len(dev), 7), dtype=np.float32)
        for k, fold in enumerate(folds, 1):
            rest = np.setdiff1d(np.arange(len(dev)), fold)
            h, model, _ = train_eval(cfg, [dev_texts[i] for i in rest], ydev[rest], [dev_texts[i] for i in fold],
                                     cfg["epochs"])
            oof[:, fold] = h
            del model
            torch.cuda.empty_cache()
            log(f"gabung: lipatan {k}/5")
        np.save(path, oof)
    per_epoch = [crossfit_f1(ydev, oof[e], folds) for e in range(cfg["epochs"])]
    best = int(np.argmax([x.mean() for x in per_epoch]))
    cv = per_epoch[best]
    log(f"gabung: CV macro-F1 {cv.mean():.3f} (epoch {best + 1}) " + " ".join(f"{c} {v:.2f}" for c, v in zip(cols, cv)))
    th = tune(ydev, oof[best])
    hist, model, _ = train_eval(cfg, dev_texts, ydev, te_texts, cfg["epochs"])
    del model
    p_te = hist[best]
    prec, rec, f1, sup = precision_recall_fscore_support(y7[te], p_te >= th, average=None, zero_division=0)
    report = {"skenario": "EN05+EN06 digabung", "epoch": best + 1, "thresholds": dict(zip(cols, th.tolist())),
              "cv_dev_macro_f1": round(float(cv.mean()), 4), "cv_dev_f1": dict(zip(cols, cv.round(3).tolist())),
              "test_macro_f1": round(float(f1.mean()), 4),
              "test": {c: {"precision": round(float(prec[j]), 3), "recall": round(float(rec[j]), 3),
                           "f1": round(float(f1[j]), 3), "test_pos": int(sup[j])} for j, c in enumerate(cols)}}
    (OUT / "final_gabung.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(OUT / "final_gabung_test_probs.npy", p_te)
    print(json.dumps(report, indent=2))


# ---------- ringkasan dan evaluasi akhir ----------
def candidates(y, dev, folds):
    """{nama: probabilitas out-of-fold dev} untuk setiap model yang sudah dijalankan (fine-tuning: epoch terbaik),
    plus "semua" = rata-rata probabilitas semua model tersebut."""
    ydev = y[dev]
    cands = {}
    for p in sorted(OUT.glob("oof_lin_*.npy")):
        cands[f"lin_{p.stem.removeprefix('oof_lin_')}"] = np.load(p)
    for p in sorted(OUT.glob("oof_ft_*.npy")):
        oof = np.load(p)
        e = int(np.argmax([crossfit_f1(ydev, oof[i], folds).mean() for i in range(len(oof))]))
        cands[f"ft_{p.stem.removeprefix('oof_ft_')}@{e + 1}"] = oof[e]
    cands["semua"] = np.mean(list(cands.values()), axis=0)
    return cands


def cmd_ringkas(texts, y, dev, te, folds):
    ydev = y[dev]
    rows = []
    for name, oof in candidates(y, dev, folds).items():
        f1 = crossfit_f1(ydev, oof, folds)
        rows.append((f1.mean(), name, f1))
    rows.sort(reverse=True)
    lines = [f"{'kandidat':<48} macro  " + "  ".join(EN_COLS)]
    for m, name, f1 in rows:
        lines.append(f"{name:<48} {m:.3f}  " + "  ".join(f"{v:.2f}" for v in f1))
    text = "\n".join(lines)
    print(text)
    (OUT / "ringkasan_cv.txt").write_text(text + "\n", encoding="utf-8")


def predict_component(comp, texts, y, dev, te):
    """Latih satu komponen kandidat pada seluruh dev, kembalikan probabilitas test."""
    dev_texts, te_texts = [texts[i] for i in dev], [texts[i] for i in te]
    if comp.startswith("lin_"):
        feat = comp.removeprefix("lin_")
        C = json.loads((OUT / f"lin_{feat}.json").read_text())["C"]
        if feat == "tfidf":
            Xtr, Xte = tfidf_features(dev_texts, te_texts)
        else:
            X = embeddings(feat, texts)
            Xtr, Xte = X[dev], X[te]
        return lr_fit_predict(Xtr, y[dev], Xte, C)
    name, epoch = comp.removeprefix("ft_").split("@")
    cfg = CONFIGS[name]
    # jadwal learning rate sama dengan CV (cfg["epochs"]), lalu diambil snapshot pada epoch terpilih
    hist, model, _ = train_eval(cfg, dev_texts, y[dev], te_texts, cfg["epochs"])
    del model
    torch.cuda.empty_cache()
    return hist[int(epoch) - 1]


def cmd_final(name, texts, y, dev, te, folds):
    cands = candidates(y, dev, folds)
    parts = [k for k in cands if k != "semua"] if name == "semua" else [name]
    th = tune(y[dev], cands[name])
    p_te = np.mean([predict_component(c, texts, y, dev, te) for c in parts], axis=0)
    yte = y[te]
    prec, rec, f1, sup = precision_recall_fscore_support(yte, p_te >= th, average=None, zero_division=0)
    cv = crossfit_f1(y[dev], cands[name], folds)
    report = {"kandidat": name, "komponen": parts, "thresholds": dict(zip(EN_COLS, th.tolist())),
              "cv_dev_macro_f1": round(float(cv.mean()), 4), "cv_dev_f1": dict(zip(EN_COLS, cv.round(3).tolist())),
              "test_macro_f1": round(float(f1.mean()), 4),
              "test_macro_f1_estimable": round(float(f1[sup >= 5].mean()), 4),
              "test": {c: {"precision": round(float(prec[j]), 3), "recall": round(float(rec[j]), 3),
                           "f1": round(float(f1[j]), 3), "test_pos": int(sup[j])} for j, c in enumerate(EN_COLS)}}
    tag = name.replace("@", "_e")
    (OUT / f"final_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(OUT / f"final_{tag}_test_probs.npy", p_te)
    print(json.dumps(report, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    texts, y, dev, te, folds = load_split()
    log(f"dev {len(dev)} | test {len(te)} | positif test {y[te].sum(0).tolist()}")
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "linear":
        cmd_linear(texts, y, dev, te, folds)
    elif cmd == "finetune":
        cmd_finetune(args or list(CONFIGS), texts, y, dev, te, folds)
    elif cmd == "dapt":
        cmd_dapt(texts, te)
    elif cmd == "ringkas":
        cmd_ringkas(texts, y, dev, te, folds)
    elif cmd == "gabung":
        cmd_gabung(texts, y, dev, te, folds)
    elif cmd == "gambar_kurva":
        cmd_gambar_kurva()
    elif cmd == "kurva":
        cmd_kurva(texts, y, dev, te, folds)
    elif cmd == "final":
        cmd_final(args[0], texts, y, dev, te, folds)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
