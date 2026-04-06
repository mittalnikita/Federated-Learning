ETTh1_CONFIG = {
    "dataset":        "ETTh1",
    "filepath":       "/kaggle/input/ett-small/ETTh1.csv",
    "n_features":     7,
    "num_clients":    10,
    "seq_len":        96,
    "pred_len":       96,
    "rounds":         100,
    "local_epochs":   5,
    "lr":             0.001,
    "client_fraction":0.3,
    "alpha_levels":   [10.0, 1.0, 0.1],
    "mu":             0.1,
    "hidden":         64,
    "layers":         2,
    "batch_size":     32,
    "seed":           42
}
