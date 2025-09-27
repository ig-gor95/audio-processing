# train_dialog_tagger.py
# ----------------------------------------------------------
# Мультилейбл-классификация тем диалога.
# Ожидает CSV с колонками: dialog_id, text, topics_csv (темы через запятую).
#
# Установка:
#   pip install scikit-learn pandas numpy scipy joblib
#
# Примеры:
#   Обучение + валидация + сохранение:
#       python train_dialog_tagger.py --train_csv dialogs_ds.csv --out_model dialog_tagger.joblib
#
#   Только инференс на тексте:
#       python train_dialog_tagger.py --load_model dialog_tagger.joblib --predict "текст диалога..."
#
#   Оценка готовой модели на CSV:
#       python train_dialog_tagger.py --load_model dialog_tagger.joblib --eval_csv dialogs_ds.csv
# ----------------------------------------------------------

import argparse
import json
import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load


# -------------------------- utils --------------------------

ALLOWED_TOPICS = {
    # продажи/оформление
    "продажа шин",
    "продажа услуг шиномонтажа",
    "продажа услуг установки дисков",
    "продажа услуг по замене масла",
    "продажа дисков",
    "продажа колес",
    "продажа аккумулятора",
    "продажа покрышек",
    "оформление возврата",
    # вопросы
    "вопрос по доставке",
    "вопрос по оформлению",
    "вопрос по шинам",
    "вопрос по колеса",
    "вопрос по дискам",
    "вопрос по возврату",
    "вопрос по аккумулятору",
    "вопрос по шиномонтажу",
    "вопрос по установке дисков",
    "вопрос по замене масла",
    "вопрос по гарантии",
    "вопрос по заказу",
    "вопрос по хранению колес",
    # жалоба
    "жалоба",
}


def _norm_topic(t: str) -> str:
    s = (t or "").strip().lower().replace("ё", "е")
    if not s:
        return ""
    if s.startswith("жалоба"):
        return "жалоба"
    return s


def parse_topics_csv(cell: str) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    parts = [p.strip() for p in str(cell).split(",")]
    out: List[str] = []
    for p in parts:
        n = _norm_topic(p)
        if not n:
            continue
        if ALLOWED_TOPICS and n not in ALLOWED_TOPICS:
            # если хочешь не фильтровать — закомментируй проверку
            continue
        out.append(n)
    # уникальные с сохранением порядка
    return list(dict.fromkeys(out))


def basic_clean(text: str) -> str:
    """Быстрая чистка: нижний регистр, ё→е, схлоп пробелов, убираем мусорные повторения."""
    if not isinstance(text, str):
        text = str(text or "")
    t = text.lower().replace("ё", "е")
    # уберём служебные префиксы спикеров, если они мешают
    t = re.sub(r'^(sales|client|speaker_[12])\s*:\s*', '', t, flags=re.IGNORECASE | re.MULTILINE)
    # схлоп пробелов
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_dataset(csv_path: str) -> Tuple[List[str], List[List[str]], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    need = {"dialog_id", "text", "topics_csv"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"В {csv_path} не хватает колонок: {sorted(miss)}")

    df["text"] = df["text"].astype(str).map(basic_clean)
    df["labels"] = df["topics_csv"].map(parse_topics_csv)
    df = df[df["labels"].map(len) > 0].reset_index(drop=True)

    texts = df["text"].tolist()
    labels = df["labels"].tolist()
    return texts, labels, df


def build_vectorizers() -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    # Слова: 1–2-граммы, урезаем редкие, русская морфология частично ловится би-граммами
    word_v = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    # Символы: 3–5-граммы, хорошо ловят опечатки/формы
    char_v = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )
    return word_v, char_v


def vectorize_fit(word_v, char_v, texts: List[str]):
    Xw = word_v.fit_transform(texts)
    Xc = char_v.fit_transform(texts)
    return hstack([Xw, Xc]).tocsr()


def vectorize_transform(word_v, char_v, texts: List[str]):
    Xw = word_v.transform(texts)
    Xc = char_v.transform(texts)
    return hstack([Xw, Xc]).tocsr()


def build_classifier() -> OneVsRestClassifier:
    # Лёгкая и точная связка: OVR + логрег с балансировкой классов
    base = LogisticRegression(
        solver="liblinear",  # стабильный на небольших данных
        class_weight="balanced",
        max_iter=200
    )
    return OneVsRestClassifier(base, n_jobs=-1)


def pick_threshold(y_true_bin: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Подбираем общий порог по валидации (максимизируем micro-F1)
    y_scores — decision_function или вероятность.
    """
    # если классификатор вернул decision_function (может быть <0..>0), приведём к [0..1] через сигмоиду
    if (y_scores.min() < 0) or (y_scores.max() > 1):
        y_prob = 1 / (1 + np.exp(-y_scores))
    else:
        y_prob = y_scores

    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true_bin, y_pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def evaluate(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, mlb: MultiLabelBinarizer) -> Dict[str, Any]:
    micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    report = classification_report(
        y_true_bin, y_pred_bin, target_names=list(mlb.classes_), zero_division=0, output_dict=True
    )
    return {"f1_micro": micro, "f1_macro": macro, "report": report}


def save_model(path: str, payload: Dict[str, Any]) -> None:
    dump(payload, path)


def load_model(path: str) -> Dict[str, Any]:
    return load(path)


def predict_labels(model: Dict[str, Any], texts: List[str]) -> List[List[str]]:
    wv = model["word_vect"]
    cv = model["char_vect"]
    clf = model["clf"]
    mlb = model["mlb"]
    thr = model["threshold"]

    X = vectorize_transform(wv, cv, [basic_clean(t) for t in texts])
    # предпочтительно decision_function (есть у логрег с liblinear)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        scores = clf.predict_proba(X)
    # нормируем к [0..1] если нужно
    if (scores.min() < 0) or (scores.max() > 1):
        probs = 1 / (1 + np.exp(-scores))
    else:
        probs = scores
    pred_bin = (probs >= thr).astype(int)
    label_lists = mlb.inverse_transform(pred_bin)
    return [list(labels) for labels in label_lists]


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="", help="CSV с колонками dialog_id,text,topics_csv")
    ap.add_argument("--out_model", type=str, default="dialog_tagger.joblib", help="куда сохранить модель")
    ap.add_argument("--val_size", type=float, default=0.15, help="доля валидации")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--load_model", type=str, default="", help="путь к .joblib для инференса/оценки")
    ap.add_argument("--predict", type=str, default="", help="один текст для предсказания")
    ap.add_argument("--eval_csv", type=str, default="", help="оценить модель на CSV (dialog_id,text,topics_csv)")

    args = ap.parse_args()

    # ---- режим инференса (без обучения)
    if args.load_model and (args.predict or args.eval_csv):
        mdl = load_model(args.load_model)

        if args.predict:
            labels = predict_labels(mdl, [args.predict])[0]
            print(", ".join(labels))
            return

        if args.eval_csv:
            texts, labels, _ = load_dataset(args.eval_csv)
            wv, cv, clf, mlb, thr = mdl["word_vect"], mdl["char_vect"], mdl["clf"], mdl["mlb"], mdl["threshold"]
            X = vectorize_transform(wv, cv, texts)
            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(X)
            else:
                scores = clf.predict_proba(X)
            if (scores.min() < 0) or (scores.max() > 1):
                probs = 1 / (1 + np.exp(-scores))
            else:
                probs = scores

            y_true = mlb.transform(labels)
            y_pred = (probs >= thr).astype(int)
            metrics = evaluate(y_true, y_pred, mlb)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return

    # ---- режим обучения
    if not args.train_csv:
        raise SystemExit("Укажи --train_csv ИЛИ --load_model вместе с --predict/--eval_csv.")

    texts, labels, df = load_dataset(args.train_csv)

    # бинaризация меток
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    # сплит
    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, y, test_size=args.val_size, random_state=args.seed, stratify=(y.sum(axis=1) > 0)
    )

    # векторизация
    word_v, char_v = build_vectorizers()
    Xtr = vectorize_fit(word_v, char_v, X_tr)
    Xva = vectorize_transform(word_v, char_v, X_val)

    # классификатор
    clf = build_classifier()
    clf.fit(Xtr, y_tr)

    # подбор порога
    if hasattr(clf, "decision_function"):
        scores_val = clf.decision_function(Xva)
    else:
        scores_val = clf.predict_proba(Xva)
    thr = pick_threshold(y_val, scores_val)

    # финальная оценка
    if (scores_val.min() < 0) or (scores_val.max() > 1):
        probs_val = 1 / (1 + np.exp(-scores_val))
    else:
        probs_val = scores_val
    y_pred_val = (probs_val >= thr).astype(int)
    metrics = evaluate(y_val, y_pred_val, mlb)

    print("Best threshold:", round(thr, 3))
    print("Validation F1 micro:", round(metrics["f1_micro"], 4), "macro:", round(metrics["f1_macro"], 4))
    # короткий отчёт по меткам
    print(json.dumps({k: v for k, v in metrics["report"].items() if k in list(mlb.classes_)[:10]}, ensure_ascii=False, indent=2))

    # сохраняем
    payload = {
        "word_vect": word_v,
        "char_vect": char_v,
        "clf": clf,
        "mlb": mlb,
        "threshold": thr,
        "allowed_topics": sorted(list(ALLOWED_TOPICS)),
        "version": 1,
    }
    save_model(args.out_model, payload)
    print(f"Saved model -> {args.out_model}")


if __name__ == "__main__":
    main()
