import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda, Add, Masking, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_nlp.layers import TransformerEncoder
import tensorflow.keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense
from custom_classes import CVE, Transformer, Attention, smart_cond

# --- Deep models development --- #

# STraTS


def build_strats(max_len, num_var, d, num_repeat, num_head, dropout):

    vars = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))

    vars_emb = Embedding(num_var+1, d)(vars)
    cve_units = int(np.sqrt(d))
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)

    comb_emb = Add(name='comb_emb')([vars_emb, values_emb, times_emb])

    mask = Lambda(lambda x: K.clip(x, 0, 1))(vars)
    cont_emb = Transformer(num_repeat, num_head, dk=None, dv=None,
                           dff=None, dropout=dropout)(comb_emb, mask=mask)
    attn_weights = Attention(2*d)(cont_emb, mask=mask)
    fused_emb = Lambda(lambda x: K.sum(
        x[0]*x[1], axis=-2))([cont_emb, attn_weights])
    output = Dense(num_var)(fused_emb)
    model = Model([times, values, vars], output)

    return model


# SERT


def build_SERT(max_len,
               num_var,
               emd_dim,
               num_head,
               ffn_dim,
               num_repeat):

    vars = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))

    cve_units = int(np.sqrt(emd_dim))
    values_emb = CVE(cve_units, emd_dim)(values)
    times_emb = CVE(cve_units, emd_dim)(times)
    vars_emb = Embedding(num_var+1, emd_dim)(vars)

    comb_emb = Add(name='comb_emb')([vars_emb, values_emb, times_emb])

    mask = Lambda(lambda x: K.clip(x, 0, 1))(vars)
    pre_masked_embedding = comb_emb * \
        tf.cast(K.expand_dims(mask, axis=-1), tf.float32)

    for i in range(num_repeat):
        cont_emb = TransformerEncoder(num_heads=num_head, intermediate_dim=ffn_dim)(
            pre_masked_embedding, padding_mask=mask)

    cont_emb = cont_emb*tf.cast(K.expand_dims(mask, axis=-1), tf.float32)
    fused_emb = Flatten()(cont_emb)
    output = Dense(num_var)(fused_emb)
    model = Model([times, values, vars], output)

    return model

# SST-ANN


def build_SST_ANN(max_len, num_var, d):

    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))
    vars = Input(shape=(max_len,))

    cve_units = int(np.sqrt(d))
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)
    vars_emb = Embedding(num_var+1, d)(vars)

    comb_emb = Add(name='comb_emb')([vars_emb, values_emb, times_emb])

    mask = Lambda(lambda x: K.clip(x, 0, 1))(vars)
    pre_masked_comb_emb = comb_emb * \
        tf.cast(K.expand_dims(mask, axis=-1), tf.float32)
    comb_emb_masked = Masking()(pre_masked_comb_emb)
    flattened_comb = Flatten(name='flat')(comb_emb_masked)
    output = Dense(num_var, name='output')(flattened_comb)
    model = Model([times, values, vars], output)

    return model

# LSTM


def build_LSTM(order, num_var):

    model = Sequential()
    model.add(LSTM(60, activation='relu', input_shape=(order, num_var)))
    model.add(Dense(num_var))

    return model


sparsity_levels = [0, 0.2, 0.4, 0.6, 0.8]
observation_len = 10  # observation window length
num_var = 16  # number of variables to be forecasted
fore_max_len = (observation_len + 1) * num_var * 2  # cap the input length
lr = 0.0005  # learning rate

# Defining loss function


def forecast_loss(y_true, y_pred, dim=num_var):
    # Mask -> y_true[:, dim:]
    return K.sum(y_true[:, dim:]*(y_true[:, :dim]-y_pred)**2, axis=-1)

# ---- Fitting Triplet encoding based models: SERT, STraTS, SST-ANN -----#

rmse_sert = []
rmse_strats = []
rmse_sst_ann = []

for fraction in tqdm(sparsity_levels):

    # load train_input
    with open('Simulation_study/data/train_input_'+str(fraction)+'.pkl', 'rb') as f:
        train_input = pickle.load(f)

    # load valid_input
    with open('Simulation_study/data/valid_input_'+str(fraction)+'.pkl', 'rb') as f:
        valid_input = pickle.load(f)

    # load train_output
    with open('Simulation_study/data/train_output_'+str(fraction)+'.pkl', 'rb') as f:
        train_output = pickle.load(f)

    # load valid_output
    with open('Simulation_study/data/valid_output_'+str(fraction)+'.pkl', 'rb') as f:
        valid_output = pickle.load(f)

    # SERT instance
    sert = build_SERT(max_len=fore_max_len,
                      num_var=num_var,
                      emd_dim=60,
                      num_head=6,
                      ffn_dim=40,
                      num_repeat=6)

    sert.compile(loss=forecast_loss, optimizer=Adam(lr))
    sert.fit(train_input,
             train_output,
             epochs=1000,
             batch_size=70,
             validation_split=0.1,
             callbacks=[EarlyStopping(
                 monitor="val_loss", patience=2, mode="min")],
             verbose=0)

    # STraTS instance
    strats = build_strats(max_len=fore_max_len,
                          num_var=num_var,
                          d=60,
                          N=6,
                          he=6,
                          V=num_var,
                          dropout=0.1)

    strats.compile(loss=forecast_loss, optimizer=Adam(lr))
    strats.fit(train_input,
               train_output,
               epochs=1000,
               batch_size=70,
               validation_split=0.1,
               callbacks=[EarlyStopping(
                   monitor="val_loss", patience=2, mode="min")],
               verbose=0)

    # SST-ANN instance
    sst_ann = build_SST_ANN(max_len=fore_max_len,
                            num_var=num_var,
                            d=60)

    sst_ann.compile(loss=forecast_loss, optimizer=Adam(lr))
    sst_ann.fit(train_input,
                train_output,
                epochs=1000,
                batch_size=70,
                validation_split=0.1,
                callbacks=[EarlyStopping(
                    monitor="val_loss", patience=2, mode="min")],
                verbose=0)

    # Evaluation
    sert_pred = sert.predict(valid_input)
    strats_pred = strats.predict(valid_input)
    sst_ann_pred = sst_ann.predict(valid_input)

    mask_values = valid_output[:, num_var:2*num_var]
    mask_values[mask_values == 0] = np.nan

    observed_values = valid_output[:, 0:num_var]
    observed = mask_values * observed_values

    sert_forecasts = mask_values * sert_pred
    strats_forecasts = mask_values * strats_pred
    sst_ann_forecasts = mask_values * sst_ann_pred

    # Calculate the RMSE and R squared
    sert_rmse = np.sqrt(np.nanmean((sert_forecasts - observed)**2))
    strats_rmse = np.sqrt(np.nanmean((strats_forecasts - observed)**2))
    st_ann_rmse = np.sqrt(np.nanmean((sst_ann_forecasts - observed)**2))

    rmse_sert.append(sert_rmse)
    rmse_strats.append(strats_rmse)
    rmse_sst_ann.append(st_ann_rmse)


# ----- Fitting LSTM ----- #

df = pd.read_csv('Simulation_study/data/simulated_data.csv')
df_base = df.drop('time', axis=1)


order = 10
num_var = 16
split_hour = 37000


def split_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


rmse_lstm = []

for fraction in tqdm(sparsity_levels):

    bool_matrix = np.random.random(df_base.shape) < fraction
    bool_matrix[0, :] = False

    df_var = df_base.mask(bool_matrix)

    # Impute the missing values using mean for each column
    df_var = df_var.fillna(method='ffill')

    # split the data into train and test based on split_hour
    train = df_var[df_var.index < split_hour]
    test = df_var[df_var.index >= split_hour]
    mask = np.logical_not(bool_matrix[df_var.index >= split_hour])
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    # LSTM data preparation
    train_X, train_y = split_sequences(train.values, order)
    test_X, test_y = split_sequences(test.values, order)

    # Fitting the lstm
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(60, activation='relu', input_shape=(order, num_var)))
    model.add(Dense(num_var))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Fit the model to the training data
#     model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=0, validation_split=0.1,
#          callbacks=[
#             EarlyStopping(monitor="val_loss", patience=3, mode="min")
#         ])

    lstm_file_name = 'lstm_sim_10h_1h_' + 'spar' + str(fraction) + '.h5'
#     model.save(lstm_file_name)
    model.load_weights('Simulation_study/fitted models/' + lstm_file_name)

    lstm_preds = model.predict(test_X)
    print('LSTM predictions are ready for sparsity = ', fraction)

    mask[mask == 0] = np.nan
    err = (test_y - lstm_preds)*mask[order:]

    # Take the mean of each column of err ignoring the nan values
    rmse = np.sqrt(np.nanmean((err)**2))

    rmse_lstm.append(rmse)


# ----- Applying Naive ----- #

rmse_naive = []

for fraction in tqdm(sparsity_levels):

    bool_matrix = np.random.random(df_base.shape) < fraction  # True = missing
    # 1st row shouldn't have missing for the ffill method to work later
    bool_matrix[0] = False

    df_var = df_base.mask(bool_matrix)

    # Impute the missing values using mean for each column
#     df_var = df_var.apply(lambda x: x.fillna(x.mean()))
    df_var = df_var.fillna(method='ffill')

    # split the data into train and test based on split_hour
    train = df_var[df_var.index < split_hour]
    test = df_var[df_var.index >= split_hour]
    mask = np.logical_not(bool_matrix[df_var.index >= split_hour])
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    steps = np.arange(0, test.shape[0] - order)

    # empty dataframe to store the forecast
    df_naive_pred = pd.DataFrame()
    df_obs = pd.DataFrame()

    for start in steps:

        end = start + order

        # Naive forecast
        naive_pred = test.iloc[end-1:end, :]

        mask_2 = mask[end:end+1, :].copy()
        observed = test.iloc[end:end+1, :]*mask_2.reshape(1, 16)

        df_naive_pred = pd.concat(
            [df_naive_pred, naive_pred], axis=0, ignore_index=True)
        df_obs = pd.concat([df_obs, observed], axis=0, ignore_index=True)

    naive_errors = df_naive_pred.values - df_obs.values
    naive_errors = naive_errors.reshape(-1)
    naive_errors = naive_errors[~np.isnan(naive_errors)]

    naive_rmse = np.sqrt(np.mean((naive_errors)**2))

    rmse_naive.append(naive_rmse)

    # ---- Plotting the results ---- #
df_sim_results = pd.DataFrame({'sparsity_level': sparsity_levels,
                               'rmse_sert': rmse_sert,
                               'rmse_sst_ann': rmse_sst_ann,
                               'rmse_strats': rmse_strats,
                               'rmse_naive': rmse_naive,
                               'rmse_lstm': rmse_lstm})

plt.plot(df_sim_results.sparsity_level,
         df_sim_results.rmse_naive, color='gold', label='Naive')
plt.scatter(df_sim_results.sparsity_level,
            df_sim_results.rmse_naive, color='gold')
plt.plot(df_sim_results.sparsity_level, df_sim_results.rmse_lstm, label='LSTM')
plt.scatter(df_sim_results.sparsity_level, df_sim_results.rmse_lstm)
plt.plot(df_sim_results.sparsity_level,
         df_sim_results.rmse_sst_ann, label='SST-ANN')
plt.scatter(df_sim_results.sparsity_level, df_sim_results.rmse_sst_ann)
plt.plot(df_sim_results.sparsity_level,
         df_sim_results.rmse_sert, color='green', label='SERT')
plt.scatter(df_sim_results.sparsity_level,
            df_sim_results.rmse_sert, color='green')
plt.plot(df_sim_results.sparsity_level,
         df_sim_results.rmse_strats, color='black', label='STraTS')
plt.scatter(df_sim_results.sparsity_level,
            df_sim_results.rmse_strats, color='black')
plt.xlabel('Sparsity level %')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('sim_results.png')

plt.show()
