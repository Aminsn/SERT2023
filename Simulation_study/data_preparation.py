import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# --- Simulating data from a Guassian process --- #
# Time steps
T = 40000

# Number of features
N = 16

# Generate a vector from a Multivariate Guassian distribution
# with mean mu and covariance matrix sigma
np.random.seed(42)

# Generate mu_0
mu_0 = np.random.uniform(0, 1, N)

# Generate a symmetric positive-semidefinite matrix
# Generate a matrix with random entries
matrix = np.random.uniform(low=-1, high=1, size=(N, N))

# Compute its Gram matrix to ensure positive-semidefiniteness
cov_matrix = matrix @ matrix.T

# Generate a random vector for the first step
mv_timeseries = mu_0.reshape((1, N))


def get_time_effect(t):

    x = np.array([10*np.sin(.005*t),
                  np.cos(0.0005*t),
                  .002*t,
                  -.002*t+10*np.sin(0.005*t),
                  5*np.sin(0.0005*t),
                  12*np.cos(0.0005*t),
                  7*np.sin(0.0005*t),
                  8*np.cos(0.0005*t),
                  2*np.sin(0.0005*t),
                  3*np.cos(0.0005*t),
                  12*np.sin(0.0005*t),
                  18*np.cos(0.0005*t),
                  4*np.sin(0.0005*t),
                  15*np.cos(0.0005*t),
                  11*np.sin(0.0005*t),
                  10*np.cos(0.0005*t)])

    return x


for t in range(1, T-1):
    mu_t = 2 + 0.4*mv_timeseries[t-1, :] + get_time_effect(t)
    mv_timeseries_new = np.random.multivariate_normal(
        mu_t, cov_matrix).reshape((1, N))
    mv_timeseries = np.vstack((mv_timeseries, mv_timeseries_new))


# --- Create a dataframe with the generated time series --- #
df = pd.DataFrame(mv_timeseries, columns=['x' + str(i) for i in range(1, N+1)])
df['time'] = np.arange(1, T)

# --- Save the dataframe --- #
df.to_csv('Simulation_study/data/simulated_data.csv', index=False)

# --- Pivot long the dataframe --- #
data = df.melt(id_vars='time', var_name='var_id',
               value_name='value').rename({'time': 'hour'}, axis=1)


def inv_list(var_list, start=0):
    """Creates a dictionary that maps each element
    in the list of variables to an integer."""

    d = {}
    for i in range(len(var_list)):
        d[var_list[i]] = i+start
    return d


# Get variable indices & convert var_id names to their indices in the dataset
varis = sorted(list(set(data.var_id)))
V = len(varis)
var_to_ind = inv_list(varis, start=1)

data['var_id'] = data['var_id'].apply(lambda x: x[1:]).astype(int)

# Normalize value by location and var_id
data['value'] = data.groupby(['var_id'])['value'].transform(
    lambda x: (x - x.mean()) / x.std())


# Split hour for training and testing sets
split_hour = 37000

# Defining sparsity levels
sparsity_levels = [0, 0.2, 0.4, 0.6, 0.8]

# Utility functions.


def create_output_array(x):
    """Concatenates the observed (& unobserved = 0)
      outputs with a masking array indicating
      which variables to be considered in the loss function."""

    mask = [0 for i in range(V)]
    values = [0 for i in range(V)]
    for vv in x:
        v = int(vv[0])-1
        mask[v] = 1
        values[v] = vv[1]
    return values+mask


def pad_input(x, n):
    """Pads the input data to allow for varying input lengths."""
    return np.concatenate([x, np.zeros(n-x.shape[0])])


hours = data.hour.values

for fraction in tqdm(sparsity_levels):

    data_sparse = data.sample(frac=1-fraction, random_state=1)

    # order data_sparse by hour and var_id
    data_sparse = data_sparse.sort_values(by=['hour', 'var_id'])

    # Hyperparameters
    pred_window_len = 1  # Forecast horizon (hours)
    obs_window_len = 10  # Observation window (hours)
    end_time = data.hour.max() - obs_window_len
    # Observed windows (Observed length, end time, sliding steps)
    start_indices = np.arange(0, end_time, pred_window_len)
    fore_max_len = (obs_window_len+1)*V*2  # Cap the input length

    hours = data_sparse.hour.values

    # Convert Pandas to Numpy to make the following computations faster
    df_neural = data_sparse.values

    # Allocating memory for the results
    results = []

    for i in range(len(set(hours))-obs_window_len+pred_window_len):

        obs_data = df_neural[(hours >= i) & (hours <= i+obs_window_len)]
        obs_data = np.hstack(
            (obs_data, np.repeat(i, len(obs_data)).reshape(-1, 1)))
        results.append(obs_data)

    results = np.concatenate(results, axis=0)
    print('check point 1')

    obs_data = pd.DataFrame(results, columns=np.concatenate(
        [data_sparse.columns.values, np.array(['window'])]))
    pred_data = obs_data.groupby('window').agg(
        {'hour': lambda x: np.max(x) + 1}).reset_index()
    pred_data = pred_data.merge(obs_data, on='hour')
    pred_data['var_id_value'] = pred_data[['var_id', 'value']].values.tolist()
    pred_data = pred_data.groupby(['window_x']).agg(
        {'var_id_value': list, 'hour': np.min}).reset_index()
    pred_data['var_id_value'] = pred_data['var_id_value'].apply(
        create_output_array)
    obs_data = obs_data.groupby(['window']).agg(
        {'var_id': list, 'hour': list, 'value': list}).reset_index()

    df_merged = obs_data.merge(pred_data,
                               left_on=['window'],
                               right_on=['window_x']).drop(['window',
                                                            'window_x'],
                                                           axis=1)
    df_merged['var_id'] = df_merged['var_id'].apply(
        lambda x: pad_input(np.array(x), fore_max_len))
    df_merged['hour_x'] = df_merged['hour_x'].apply(
        lambda x: pad_input(np.array(x), fore_max_len))
    df_merged['value'] = df_merged['value'].apply(
        lambda x: pad_input(np.array(x), fore_max_len))
    print('check point 2')

    # To be used for performance evaluation
    query_hours = df_merged['hour_y'].values

    var_id_inputs = np.array(df_merged['var_id'].values.tolist())
    hour_input = np.array(df_merged['hour_x'].values.tolist())
    values_input = np.array(df_merged['value'].values.tolist())

    # Output: Forecast values
    output = np.array(df_merged['var_id_value'].values.tolist())

    all_triplets = [hour_input, values_input, var_id_inputs]

    # Spliting train and valid.
    # Calculate each row's maximum value of hour_input.
    hour = np.array(df_merged['hour_x'].values.tolist())

    max_hour = np.max(hour, axis=1)

    train_idx = np.where(max_hour < split_hour)
    valid_idx = np.where(max_hour >= split_hour)

    # Inputs
    train_input = [x[train_idx] for x in all_triplets]
    valid_input = [x[valid_idx] for x in all_triplets]

    # Outputs
    train_output = output[train_idx]
    valid_output = output[valid_idx]

    # save train_input in data folder using pickle
    with open('Simulation_study/data/train_input_'+str(fraction)+'.pkl', 'wb') as f:
        pickle.dump(train_input, f)

    # save valid_input
    with open('Simulation_study/data/valid_input_'+str(fraction)+'.pkl', 'wb') as f:
        pickle.dump(valid_input, f)

    # save train_output
    with open('Simulation_study/data/train_output_'+str(fraction)+'.pkl', 'wb') as f:
        pickle.dump(train_output, f)

    # save valid_output
    with open('Simulation_study/data/valid_output_'+str(fraction)+'.pkl', 'wb') as f:
        pickle.dump(valid_output, f)
