CONFIG = {
    'dataset'         : 'DJIA10',    # 10 selected DJIA stocks
    'n_features'      : 10,          # 10 stocks = 10 features
    'num_clients'     : 10,          # one client per stock
    'seq_len'         : 60,          # 60 trading days lookback
    'pred_len'        : 20,          # predict next 20 days
    'rounds'          : 100,
    'local_epochs'    : 5,
    'lr'              : 0.0005,
    'client_fraction' : 0.3,
    'alpha_levels'    : [10.0, 1.0, 0.1],
    'mu'              : 0.1,
    'hidden'          : 64,
    'layers'          : 2,
    'batch_size'      : 32,
    'seed'            : 42
}
