import numpy as np
import pandas as pd


LEARNING_RATE = 0.05
EPOCHS = 3000
L2_REG = 0.05
TEST_SIZE = 0.2
VAL_SIZE = 0.2
POS_WEIGHT_MODE = "auto" 

SEEDS = [0, 1, 2, 3, 4, 5, 10, 22, 42, 99]
THRESHOLDS = np.linspace(0.1, 0.9, 20)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def standardize(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (val - mean) / std, (test - mean) / std

def stratified_split_indices(y, test_size, seed):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(len(idx0) * test_size)
    n1_test = int(len(idx1) * test_size)

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx

def confusion(y_true, y_pred):
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    return TN, FP, FN, TP

def metrics_from_conf(TN, FP, FN, TP):
    #для класса 1
    acc = (TN + TP) / max((TN + FP + FN + TP), 1)

    recall_1 = TP / max((TP + FN), 1)
    precision_1 = TP / max((TP + FP), 1)
    f1_1 = 2 * precision_1 * recall_1 / max((precision_1 + recall_1), 1e-12)

    #для класса 0
    recall_0 = TN / max((TN + FP), 1)
    precision_0 = TN / max((TN + FN), 1)
    f1_0 = 2 * precision_0 * recall_0 / max((precision_0 + recall_0), 1e-12)

    return acc, recall_1, precision_1, f1_1, recall_0, precision_0, f1_0

def train_logreg(X, y, l2_reg, lr, epochs, pos_weight):
    n, d = X.shape
    w = np.zeros(d)

    if pos_weight == "auto":
        n_pos = max(int(y.sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        pw = n_neg / n_pos
    else:
        pw = float(pos_weight)

    sample_w = np.where(y == 1, pw, 1.0)

    for ep in range(epochs):
        p = sigmoid(X @ w)
        err = (p - y) * sample_w
        grad = (X.T @ err) / n

        if l2_reg > 0:
            grad[1:] += l2_reg * w[1:]

        w -= lr * grad

    return w, pw

def pick_threshold_on_val(probs, y_val):
    best = None
    for t in THRESHOLDS:
        pred = (probs >= t).astype(int)
        TN, FP, FN, TP = confusion(y_val, pred)
        acc, rec1, prec1, f11, rec0, prec0, f10 = metrics_from_conf(TN, FP, FN, TP)

        if (best is None) or (f11 > best["f1_1"]):
            best = {
                "t": float(t),
                "TN": TN, "FP": FP, "FN": FN, "TP": TP,
                "acc": acc,
                "rec_1": rec1, "prec_1": prec1, "f1_1": f11,
                "rec_0": rec0, "prec_0": prec0, "f1_0": f10
            }
    return best


#ОСНОВНОЙ ЦИКЛ
def run_one_seed(seed):
    df = pd.read_csv("data_f.csv")
    y = df["Attrition"].to_numpy(dtype=int)
    X = df.drop(columns=["Attrition"]).to_numpy(dtype=float)

    train_idx, test_idx = stratified_split_indices(y, TEST_SIZE, seed)
    X_trainval, y_trainval = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_idx2, val_idx2 = stratified_split_indices(y_trainval, VAL_SIZE, seed + 999)
    X_train, y_train = X_trainval[train_idx2], y_trainval[train_idx2]
    X_val, y_val = X_trainval[val_idx2], y_trainval[val_idx2]

    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    X_train = add_bias(X_train)
    X_val = add_bias(X_val)
    X_test = add_bias(X_test)

    w, pw = train_logreg(X_train, y_train, L2_REG, LEARNING_RATE, EPOCHS, POS_WEIGHT_MODE)

    val_probs = sigmoid(X_val @ w)
    best = pick_threshold_on_val(val_probs, y_val)
    thr = best["t"]

    test_probs = sigmoid(X_test @ w)
    test_pred = (test_probs >= thr).astype(int)
    TN, FP, FN, TP = confusion(y_test, test_pred)

    acc, rec1, prec1, f11, rec0, prec0, f10 = metrics_from_conf(TN, FP, FN, TP)

    return {
        "seed": seed,
        "pos_weight_used": pw,
        "thr": thr,
        "test_size": len(y_test),
        "test_pos": int(y_test.sum()),
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "acc": acc,
        "rec_1": rec1, "prec_1": prec1, "f1_1": f11,
        "rec_0": rec0, "prec_0": prec0, "f1_0": f10
    }

def main():
    print(f"TEST_SIZE={TEST_SIZE}, VAL_SIZE={VAL_SIZE}, L2_REG={L2_REG}, EPOCHS={EPOCHS}")
    print(f"POS_WEIGHT_MODE={POS_WEIGHT_MODE}")
    print("-" * 50)

    results = []
    for s in SEEDS:
        r = run_one_seed(s)
        results.append(r)

        print(
            f"seed={r['seed']:>3} | test: n={r['test_size']}, pos={r['test_pos']} | "
            f"thr={r['thr']:.2f} pw={r['pos_weight_used']:.2f} | "
            f"acc={r['acc']:.3f} "
            f"rec1={r['rec_1']:.3f} prec1={r['prec_1']:.3f} f1_1={r['f1_1']:.3f} | "
            f"rec0={r['rec_0']:.3f} prec0={r['prec_0']:.3f} f1_0={r['f1_0']:.3f} | "
            f"TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}"
        )

    accs = np.array([r["acc"] for r in results])

    rec1s = np.array([r["rec_1"] for r in results])
    prec1s = np.array([r["prec_1"] for r in results])
    f11s = np.array([r["f1_1"] for r in results])

    rec0s = np.array([r["rec_0"] for r in results])
    prec0s = np.array([r["prec_0"] for r in results])
    f10s = np.array([r["f1_0"] for r in results])

    print("-" * 50)
    print("СРЕДНЕЕ ПО SEED (± std):")
    print(f"Accuracy:  {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"Recall(1): {rec1s.mean():.3f} ± {rec1s.std():.3f}")
    print(f"Prec(1):   {prec1s.mean():.3f} ± {prec1s.std():.3f}")
    print(f"F1(1):     {f11s.mean():.3f} ± {f11s.std():.3f}")
    print(f"Recall(0): {rec0s.mean():.3f} ± {rec0s.std():.3f}")
    print(f"Prec(0):   {prec0s.mean():.3f} ± {prec0s.std():.3f}")
    print(f"F1(0):     {f10s.mean():.3f} ± {f10s.std():.3f}")

if __name__ == "__main__":
    main()
