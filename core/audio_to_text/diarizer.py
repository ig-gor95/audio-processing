import os
os.environ.setdefault("PYTORCH_ENABLE_NNPACK", "0")  # избегаем NNPack-конфликтов

from pathlib import Path
import math
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference import EncoderClassifier
from silero_vad import load_silero_vad, get_speech_timestamps

from core.dto.diarisation_result import DiarizedResult
from log_utils import setup_logger
from yaml_reader import ConfigLoader

# ================= Setup =================
logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

SAMPLE_RATE   = int(config.get('audio.sample_rate', 16000))
VAD_THR       = float(config.get('diarize.vad_threshold', 0.45))
MAX_SPEAKERS  = int(config.get('diarize.max_speakers', 2))
FORCE_TWO     = bool(config.get('diarize.force_two', True))
BATCH         = int(config.get('diarize.batch_size', 64))

# окно/шаг для эмбеддингов (повышаем разрешение по времени)
WIN_SEC       = float(config.get('diarize.win_sec', 1.0))
HOP_SEC       = float(config.get('diarize.hop_sec', 0.30))

# порог склейки соседних окон одного спикера, мин. длительность фин. сегмента
MERGE_GAP     = float(config.get('diarize.merge_gap', 0.15))
MIN_KEEP      = float(config.get('diarize.min_keep', 1.0))

# сглаживание меток и минимальный «заход»
MEDIAN_K      = int(config.get('diarize.median_k', 3))
MIN_DWELL     = float(config.get('diarize.min_dwell', 1.0))

# округление таймингов — мягкое (мелкий шаг, расширяющее)
ROUND_Q       = float(config.get('diarize.round_q', 0.05))

# Viterbi/маржин
TAU_MARGIN      = float(config.get('diarize.tau_margin', 0.025))
SWITCH_PENALTY  = float(config.get('diarize.switch_penalty', 0.20))

# смещение разреза в сторону правого сегмента, чтобы не резать хвост левого
CUT_BIAS        = float(config.get('diarize.cut_bias', 0.04))  # сек

# НОВОЕ: принудительные разрезы внутри длинных «однолейбловых» участков
TAU_STRONG        = float(config.get('diarize.tau_strong', 0.12))   # насколько уверенно выигрывает другой спикер
HOLD_STRONG_SEC   = float(config.get('diarize.hold_strong_sec', 0.6))# минимум длительности такого выигрыша
HYSTERESIS        = float(config.get('diarize.hysteresis', 0.02))    # чтобы не «пилить» границу

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# ================= Utils =================
def read_audio_with_sr(file_path: Path):
    """Read audio and return mono FloatTensor [T] at SAMPLE_RATE."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(file_path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)  # mono [T]
        return wav, SAMPLE_RATE
    except Exception as e:
        logger.error(f"Audio loading failed: {e}")
        raise

def merge_close(ts, gap=0.20):
    """Склейка близких VAD-сегментов (сек)."""
    out = []
    for seg in ts:
        s = float(seg['start']); e = float(seg['end'])
        if not out or s - out[-1]['end'] > gap:
            out.append({'start': s, 'end': e})
        else:
            out[-1]['end'] = max(out[-1]['end'], e)
    return out

def cmvn(x: np.ndarray) -> np.ndarray:
    m, s = float(x.mean()), float(x.std())
    if not np.isfinite(s) or s < 1e-6: s = 1.0
    return (x - m) / s

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def round_segments_expand(segs, q: float = ROUND_Q, total_dur: float | None = None):
    """
    Расширяющее округление: start -> floor(q), end -> ceil(q); перекрытия отдаём правому старту.
    """
    if not segs:
        return segs
    r = []
    for s in segs:
        a = max(0.0, math.floor(float(s['start']) / q) * q)
        b = math.ceil(float(s['end']) / q) * q
        if total_dur is not None:
            b = min(b, total_dur)
        r.append({'start': a, 'end': max(a, b)})
    r.sort(key=lambda z: z['start'])
    for i in range(1, len(r)):
        if r[i]['start'] < r[i-1]['end'] - 1e-9:
            r[i]['start'] = r[i-1]['end']
        if r[i]['end'] < r[i]['start']:
            r[i]['end'] = r[i]['start']
    return r

def pad_segments_for_asr(segs, pad_left=0.12, pad_right=0.18, total_dur: float | None = None):
    """
    «Подушка» до/после сегмента без налезания на соседей.
    """
    if not segs:
        return segs
    out = []
    for i, s in enumerate(segs):
        a = s['start'] - pad_left
        b = s['end'] + pad_right
        if i > 0:
            a = max(a, segs[i-1]['end'])
        a = max(0.0, a)
        if total_dur is not None:
            b = min(b, total_dur)
        if i < len(segs) - 1:
            b = min(b, segs[i+1]['start'])
        out.append({'start': max(a, s['start']), 'end': max(max(a, s['start']), b)})
    return out

# ================= Windowing =================
def windows_over_segments(vad_segments, win_sec=WIN_SEC, hop_sec=HOP_SEC, min_len=MIN_KEEP):
    """
    Фиксированные окна внутри каждого VAD-сегмента; храним vad_id.
    """
    out = []
    for idx, seg in enumerate(vad_segments):
        s, e = float(seg['start']), float(seg['end'])
        dur = e - s
        if dur < min_len - 1e-6:
            continue
        if dur <= win_sec + 1e-6:
            out.append({'start': s, 'end': e, 'vad_id': idx})
            continue
        t = s
        while t + win_sec <= e + 1e-9:
            out.append({'start': t, 'end': min(t + win_sec, e), 'vad_id': idx})
            t += hop_sec
    return out

# ================= Label helpers =================
def median_filter_labels(labels: np.ndarray, k=MEDIAN_K) -> np.ndarray:
    if k <= 1 or len(labels) == 0:
        return labels
    k = int(k) if k % 2 == 1 else int(k) + 1
    rad = k // 2
    y = labels.copy()
    for i in range(len(labels)):
        a = max(0, i - rad)
        b = min(len(labels), i + rad + 1)
        y[i] = np.bincount(labels[a:b]).argmax()
    return y

def centroids_from_labels(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """L2-нормированные центроиды; если кластер пуст — берём самый «дальний» от среднего."""
    C = np.zeros((k, X.shape[1]), dtype=np.float32)
    gmean = X.mean(0)
    for c in range(k):
        idx = (labels == c)
        if idx.any():
            vec = X[idx].mean(0)
        else:
            j = int(np.argmax(np.linalg.norm(X - gmean, axis=1)))
            vec = X[j]
        n = np.linalg.norm(vec) + 1e-12
        C[c] = vec / n
    return C

def viterbi_group(cost: np.ndarray, penalty: float = SWITCH_PENALTY) -> np.ndarray:
    """Viterbi для одной группы окон: cost[N,K] = -cos_sim, штраф за переключение."""
    N, K = cost.shape
    dp = np.zeros_like(cost, dtype=np.float32)
    bk = np.zeros((N, K), dtype=np.int32)
    dp[0] = cost[0]
    I = np.eye(K, dtype=np.float32)
    for i in range(1, N):
        trans = dp[i-1][:, None] + penalty * (1.0 - I)
        bk[i] = trans.argmin(axis=0)
        dp[i] = cost[i] + trans.min(axis=0)
    y = np.zeros(N, dtype=np.int32)
    y[-1] = dp[-1].argmin()
    for i in range(N-2, -1, -1):
        y[i] = bk[i+1, y[i+1]]
    return y

def enforce_min_dwell_group(labels_g: np.ndarray, windows_g: list, min_dwell=MIN_DWELL) -> np.ndarray:
    """Минимальная длительность «захода» внутри одной VAD-группы."""
    if len(labels_g) == 0:
        return labels_g
    y = labels_g.copy()

    def runs_from(y):
        runs = []
        cur = y[0]; a = 0
        for i in range(1, len(y)):
            if y[i] != cur:
                runs.append((cur, a, i-1))
                cur = y[i]; a = i
        runs.append((cur, a, len(y)-1))
        return runs

    guard, changed = 0, True
    while changed and guard < 5:
        changed = False
        guard += 1
        runs = runs_from(y)
        if len(runs) <= 1:
            break
        for idx, (lab, a, b) in enumerate(runs):
            st = float(windows_g[a]['start']); en = float(windows_g[b]['end'])
            if (en - st) >= (min_dwell - 1e-6):
                continue
            # переливаем к соседу с большей длительностью
            left_dur = -1.0
            right_dur = -1.0
            if idx - 1 >= 0:
                la, lb = runs[idx-1][1], runs[idx-1][2]
                ls = float(windows_g[la]['start']); le = float(windows_g[lb]['end'])
                left_dur = le - ls
            if idx + 1 < len(runs):
                ra, rb = runs[idx+1][1], runs[idx+1][2]
                rs = float(windows_g[ra]['start']); re = float(windows_g[rb]['end'])
                right_dur = re - rs
            target_lab = y[a-1] if left_dur >= right_dur and a-1 >= 0 else (y[b+1] if b+1 < len(y) else y[a])
            y[a:b+1] = target_lab
            changed = True
    return y

def refine_by_viterbi_and_margin(windows: list, X: np.ndarray, labels_init: np.ndarray,
                                 n_clusters: int, switch_penalty: float = SWITCH_PENALTY,
                                 min_dwell: float = MIN_DWELL, tau: float = TAU_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (labels, sims), где sims = X @ C^T (L2-нормированные косинусные похожести).
    """
    C = centroids_from_labels(X, labels_init, n_clusters)
    sims = X @ C.T                     # [N,K], X и C — L2-норм
    cost = -sims

    # индексы окон по vad_id в хронологическом порядке
    from collections import defaultdict
    groups = defaultdict(list)
    for i, w in enumerate(windows):
        groups[int(w.get('vad_id', -1))].append(i)

    y = labels_init.copy()
    for gid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: float(windows[i]['start']))
        if not idxs:
            continue
        g_cost = cost[idxs]           # [M,K]
        g_sims = sims[idxs]           # [M,K]
        g_y = viterbi_group(g_cost, penalty=switch_penalty)

        # margin: если уверенность мала — не переключаемся
        top2 = np.partition(g_sims, -2, axis=1)[:, -2:]
        top2_sorted = np.sort(top2, axis=1)
        margins = top2_sorted[:, 1] - top2_sorted[:, 0]
        for j in range(1, len(idxs)):
            if margins[j] < tau:
                g_y[j] = g_y[j-1]

        # enforce min dwell внутри этого VAD-сегмента
        sub_windows = [windows[i] for i in idxs]
        g_y = enforce_min_dwell_group(g_y, sub_windows, min_dwell=min_dwell)

        y[idxs] = g_y
    return y, sims

# НОВОЕ: принудительные разрезы внутри длинных участков одного лейбла
def force_splits_by_strong_opposition(
    windows: list,
    labels: np.ndarray,
    sims: np.ndarray,
    hop_sec: float = HOP_SEC,
    tau_strong: float = TAU_STRONG,
    hold_strong_sec: float = HOLD_STRONG_SEC,
    hysteresis: float = HYSTERESIS
) -> np.ndarray:
    """
    Если внутри длинного «монолитного» участка меток обнаружен непрерывный блок окон,
    где другой спикер уверенно (>= tau_strong) выигрывает Top-1, и длительность блока >= hold_strong_sec,
    то принудительно меняем метку в этом блоке (вставляя переключение спикеров).
    """
    y = labels.copy()
    K = sims.shape[1]
    if K < 2 or len(y) == 0:
        return y

    need = max(1, int(round(hold_strong_sec / max(1e-6, hop_sec))))

    # пробегаем непрерывные ран-ы одинаковой метки
    i = 0
    N = len(y)
    while i < N:
        j = i
        while j + 1 < N and y[j + 1] == y[i]:
            j += 1
        # ран: [i..j], текущая метка c
        c = y[i]
        # для K=2 альтернативная метка — 1-c, для K>2 — argmax по k!=c
        alt = None
        if K == 2:
            alt = 1 - c

        # строим булев массив «сильно за другого»
        strong = []
        for t in range(i, j + 1):
            # current score = sims[t, c]; best other = max_k!=c sims[t,k]
            if K == 2:
                best_other = sims[t, alt]
            else:
                best_other = np.max(np.concatenate([sims[t, :c], sims[t, c+1:]], axis=0))
                alt = int(np.argmax(np.concatenate([sims[t, :c], sims[t, c+1:]], axis=0)))
            margin = best_other - sims[t, c]
            strong.append(margin >= tau_strong)
        strong = np.array(strong, dtype=bool)

        # ищем длинные непрерывные блоки strong==True
        t0 = i
        k = i
        while k <= j:
            if not strong[k - i]:
                k += 1
                continue
            # найден старт блока
            b = k
            while k <= j and strong[k - i]:
                k += 1
            e = k - 1
            if (e - b + 1) >= need:
                # расширим за счёт гистерезиса — окна, где альтернативный спикер почти выигрывает
                bb = b
                while bb - 1 >= i:
                    margin_prev = (sims[bb - 1, alt] - sims[bb - 1, c])
                    if margin_prev >= (tau_strong - hysteresis):
                        bb -= 1
                    else:
                        break
                ee = e
                while ee + 1 <= j:
                    margin_next = (sims[ee + 1, alt] - sims[ee + 1, c])
                    if margin_next >= (tau_strong - hysteresis):
                        ee += 1
                    else:
                        break
                # меняем метку внутри [bb..ee] на альтернативную
                y[bb:ee + 1] = alt
            # продолжаем поиск после e
        i = j + 1

    return y

# ================= Точная локализация границ и склейка =================
def segments_from_labeled_windows(
    windows: list,
    labels: np.ndarray,
    max_gap: float = MERGE_GAP,
    min_len: float = MIN_KEEP,
    cut_bias: float = CUT_BIAS
):
    if not windows or len(windows) != len(labels):
        return [], np.array([], dtype=int)

    items = [
        {
            'start': float(w['start']),
            'end':   float(w['end']),
            'mid':   0.5 * (float(w['start']) + float(w['end'])),
            'lab':   int(l),
            'vad_id': int(w.get('vad_id', -1)),
        }
        for w, l in zip(windows, labels)
    ]
    items.sort(key=lambda z: (z['start'], z['end']))

    segs = []
    cur = {'start': items[0]['start'], 'end': items[0]['end'], 'lab': items[0]['lab'], 'vad_id': items[0]['vad_id']}

    for i in range(1, len(items)):
        it = items[i]
        same_vad = (cur['vad_id'] == it['vad_id'])
        same_lab = (cur['lab'] == it['lab'])
        close_enough = (it['start'] - cur['end']) <= max_gap + 1e-9

        if same_vad and same_lab and close_enough:
            cur['end'] = max(cur['end'], it['end'])
            continue

        if same_vad and (not same_lab) and close_enough:
            cur_mid = 0.5 * (cur['start'] + cur['end'])
            cut = 0.5 * (cur_mid + it['mid']) + cut_bias
            cut = min(max(cut, cur['start']), it['end'])
            cut = max(min(cut, it['start']), min(cur['end'], cut))
            segs.append({'start': cur['start'], 'end': cut, 'lab': cur['lab']})
            cur = {'start': cut, 'end': it['end'], 'lab': it['lab'], 'vad_id': it['vad_id']}
            continue

        segs.append({'start': cur['start'], 'end': cur['end'], 'lab': cur['lab']})
        cur = {'start': it['start'], 'end': it['end'], 'lab': it['lab'], 'vad_id': it['vad_id']}

    segs.append({'start': cur['start'], 'end': cur['end'], 'lab': cur['lab']})

    # фильтрация коротышей (оставляем только >= min_len)
    keep = [s for s in segs if (s['end'] - s['start']) >= (min_len - 1e-6)]
    if not keep:
        return [], np.array([], dtype=int)

    out = [keep[0]]
    for s in keep[1:]:
        if s['lab'] == out[-1]['lab'] and (s['start'] - out[-1]['end']) <= max_gap + 1e-9:
            out[-1]['end'] = max(out[-1]['end'], s['end'])
        else:
            out.append(s)

    segs_final = [{'start': s['start'], 'end': s['end']} for s in out]
    labs_final = np.array([s['lab'] for s in out], dtype=int)
    return segs_final, labs_final

# ================= Embeddings =================
def extract_embeddings(
    wav_np: np.ndarray,
    segments,
    model: EncoderClassifier,
    target_len_samples: int
):
    """
    Извлекаем эмбеддинги по сегментам/окнам (ключи: start/end).
    """
    buf, kept = [], []
    for seg in segments:
        s = int(float(seg['start']) * SAMPLE_RATE)
        e = int(float(seg['end'])   * SAMPLE_RATE)
        if e <= s:
            continue
        x = wav_np[s:e].astype(np.float32, copy=False)
        # паддинг/кроп до target_len_samples
        if len(x) < target_len_samples:
            reps = int(np.ceil(target_len_samples / max(1, len(x))))
            x = np.tile(x, reps)[:target_len_samples]
        elif len(x) > target_len_samples:
            off = (len(x) - target_len_samples) // 2
            x = x[off:off + target_len_samples]
        x = cmvn(x).astype(np.float32, copy=False)
        buf.append(x); kept.append(seg)

    if not buf:
        return np.empty((0,), dtype=np.float32), []

    X_list = []
    model.eval(); torch.set_grad_enabled(False)
    with torch.no_grad():
        for i in range(0, len(buf), BATCH):
            chunk = np.stack(buf[i:i + BATCH], axis=0).astype(np.float32, copy=False)
            x_t = torch.from_numpy(chunk).contiguous()
            try:
                x_dev = x_t.to(DEVICE, dtype=torch.float32, non_blocking=True).contiguous()
                emb_t = model.encode_batch(x_dev)
                emb_np = emb_t.detach().to("cpu").contiguous().numpy()
            except Exception as e:
                if any(k in str(e) for k in ("NNPack", "MPS", "convolutionOutput", "mps")):
                    logger.warning(f"Embeddings fallback to CPU due to: {e}")
                    model.to("cpu").eval()
                    x_cpu = x_t.to("cpu", dtype=torch.float32).contiguous()
                    emb_t = model.encode_batch(x_cpu)
                    emb_np = emb_t.detach().to("cpu").contiguous().numpy()
                    try: model.to(DEVICE).eval()
                    except Exception: pass
                else:
                    raise

            if emb_np.ndim == 3 and emb_np.shape[1] == 1:
                emb_np = emb_np.squeeze(1)
            elif emb_np.ndim == 1:
                emb_np = emb_np[None, :]
            if emb_np.size:
                X_list.append(emb_np.astype(np.float32, copy=False))

    if not X_list:
        return np.empty((0,), dtype=np.float32), []
    X = np.vstack(X_list).astype(np.float32, copy=False)
    X = l2norm_rows(X)
    return X, kept

def rebalance_if_collapsed(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Если один класс <15% → принудительно ребаланс по проекции на (c1-c0)."""
    y = labels.copy()
    p0 = (y == 0).mean()
    if p0 < 0.15 or p0 > 0.85:
        if (y == 0).any() and (y == 1).any():
            c0 = X[y == 0].mean(0); c1 = X[y == 1].mean(0)
        else:
            i0 = 0
            sims = X @ X[i0] / ((np.linalg.norm(X, axis=1) * np.linalg.norm(X[i0])) + 1e-12)
            i1 = int(np.argmin(sims))
            c0, c1 = X[i0], X[i1]
        v = c1 - c0; v /= (np.linalg.norm(v) + 1e-12)
        order = np.argsort(X @ v)
        m = len(X) // 2
        y = np.zeros_like(y); y[order[m:]] = 1
    return y

# ================= Main =================
def diarize(file_path: Path) -> DiarizedResult:
    logger.info(f"Diarizing {file_path.name}")

    # 1) модели
    try:
        embedding_model = EncoderClassifier.from_hparams(
            source=config.get("diarize.model_name", "speechbrain/spkrec-ecapa-voxceleb"),
            savedir="../../pretrained_models",
            run_opts={"device": str(DEVICE)}
        )
        for m in getattr(embedding_model, "mods", {}).values():
            m.to(DEVICE)
        embedding_model.eval()
        torch.set_grad_enabled(False)
        vad_model = load_silero_vad()  # CPU
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return DiarizedResult([], np.array([]))

    # 2) аудио + VAD
    try:
        audio, _ = read_audio_with_sr(file_path)   # torch.Tensor [T], SAMPLE_RATE
        wav = audio.unsqueeze(0).cpu().contiguous()  # [1, T]
        speech_ts = get_speech_timestamps(wav, vad_model, return_seconds=True, threshold=VAD_THR)
        if not speech_ts:
            logger.warning("VAD returned 0 speech segments")
            return DiarizedResult([], np.array([]))
        speech_ts = merge_close(speech_ts, gap=0.2)
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return DiarizedResult([], np.array([]))

    # 3) окна поверх VAD
    windows = windows_over_segments(speech_ts, win_sec=WIN_SEC, hop_sec=HOP_SEC, min_len=MIN_KEEP)
    if not windows:
        logger.warning("No frames after windowing")
        return DiarizedResult([], np.array([]))

    # 4) эмбеддинги по окнам
    wav_np = audio.cpu().numpy().astype(np.float32)
    win_samples = int(WIN_SEC * SAMPLE_RATE)
    X, kept_windows = extract_embeddings(wav_np, windows, embedding_model, win_samples)
    if X.size == 0 or len(kept_windows) == 0:
        logger.warning("No valid windows after embedding")
        return DiarizedResult([], np.array([]))

    # 5) кластеризация
    try:
        n_clusters = 2 if (FORCE_TWO or MAX_SPEAKERS <= 2) else min(MAX_SPEAKERS, max(2, len(X) // 3))
        cl = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels0 = cl.fit_predict(X)
        if n_clusters == 2:
            labels0 = rebalance_if_collapsed(labels0, X)
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return DiarizedResult([], np.array([]))

    # 6) Viterbi + margin + min_dwell
    labels_refined, sims = refine_by_viterbi_and_margin(
        kept_windows, X, labels0, n_clusters,
        switch_penalty=SWITCH_PENALTY, min_dwell=MIN_DWELL, tau=TAU_MARGIN
    )

    # 7) медианное сглаживание
    labels_refined = median_filter_labels(labels_refined, k=MEDIAN_K)

    # 8) НОВОЕ: «спасательные» принудительные разрезы внутри монолитов
    labels_refined = force_splits_by_strong_opposition(
        kept_windows, labels_refined, sims,
        hop_sec=HOP_SEC, tau_strong=TAU_STRONG, hold_strong_sec=HOLD_STRONG_SEC, hysteresis=HYSTERESIS
    )

    # 9) точная локализация границ + финальное склеивание
    final_segments, final_labels = segments_from_labeled_windows(
        kept_windows, labels_refined, max_gap=MERGE_GAP, min_len=MIN_KEEP, cut_bias=CUT_BIAS
    )
    if len(final_segments) == 0:
        logger.warning("No segments after boundary localization")
        return DiarizedResult([], np.array([]))

    # 10) округление и подушка
    total_dur_sec = float(audio.shape[-1]) / SAMPLE_RATE
    final_segments = round_segments_expand(final_segments, q=ROUND_Q, total_dur=total_dur_sec)
    final_segments = pad_segments_for_asr(final_segments, pad_left=0.12, pad_right=0.18, total_dur=total_dur_sec)

    logger.info(
        f"Diarization complete. Windows: {len(kept_windows)} → Segments: {len(final_segments)}, "
        f"Speakers(forced): {len(np.unique(final_labels))}"
    )
    return DiarizedResult(final_segments, final_labels)
