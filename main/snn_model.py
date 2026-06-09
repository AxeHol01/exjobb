# imports
import os
import logging

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen, functional
from snntorch import utils
from datetime import datetime, timedelta

import torch
from torch import nn, optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd

from plotting_module import plot_results  # Import plotting function
from nab_scoring import calculate_nab_score_with_window_based_tp_fn  # Import NAB Scoring function
from anomaly_likelihood import detect_anomalies_with_likelihood, compute_anomaly_likelihood  # import Anomaly Likelihood functions
from preprocessing import create_windowed_data, clean_training_data, load_and_prepare_data, add_time_features, filter_anomaly_windows
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from scipy.stats import norm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        #encoder https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
        self.encoder = nn.Sequential(
            nn.Linear(2410, 128),
            snn.Leaky(beta=0.95, init_hidden=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            snn.Leaky(beta=0.95, init_hidden=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 14),
            snn.Leaky(beta=0.95, init_hidden=True),
            # nn.Linear(64, 36),
            # snn.Leaky(beta=0.95, init_hidden=True),
            # nn.Linear(36, 18),
            # snn.Leaky(beta=0.95, init_hidden=True),
            # nn.Linear(18, 9)

            # nn.Linear(128, 64),
            # snn.Leaky(beta=0.9),
            # nn.Linear(64, 14),
            # snn.Leaky(beta=0.9)
        )

        self.decoder = nn.Sequential(
            nn.Linear(14, 64),
            snn.Leaky(beta=0.95, init_hidden=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            snn.Leaky(beta=0.95, init_hidden=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 2410),
            snn.Leaky(beta=0.95, init_hidden=True),
            # nn.Linear(36, 64),
            # snn.Leaky(beta=0.95, init_hidden=True),
            # nn.Linear(64, 128),
            # snn.Leaky(beta=0.95, init_hidden=True),
            # nn.Linear(128, 28 * 28),
            nn.Sigmoid()
            # nn.Linear(14, 64),
            # snn.Leaky(beta=0.9),
            # nn.Linear(64, 128),
            # snn.Leaky(beta=0.9),
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        return decoded

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
criterion = nn.MSELoss()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
dtype = torch.float64
batch_size = 32
train_set = pd.read_csv("C:\\schoolstuff\\thesis\\data\\5xx_feature_dir\\train.csv")

anomaly_threshold = 0.996
long_window = 30
short_window = 2
nab_scoring_profile = "reward_fn"
#nab_scoring_profile = "standard"
#train_set = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# print(train_set)
# train_set = torch.tensor(train_set.to_numpy())

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader("C:\schoolstuff\\thesis2fold\data\\5xx_feature_dir\\test.csv",  batch_size=batch_size, shuffle=True, drop_last=True)

def training(train_data, anomaly_windows, test_data):
    #train_loader = DataLoader("C:\schoolstuff\\thesis2fold\data\\5xx_feature_dir\\filtered_df_.csv")
    
    #MOVE RECONSTRUCTION ERROR OUT OF THE TRAINING LOOP TO THE TESTING.


    epochs = 50
    loss_hist = []
    test_loss_hist = []
    counter = 0
    num_steps = 20
    batch_size = 32

    anomaly_windows['anomaly_window_start'] = anomaly_windows['anomaly_window_start'].dt.tz_localize(None)
    anomaly_windows['anomaly_window_end'] = anomaly_windows['anomaly_window_end'].dt.tz_localize(None)
    cleaned_training_data = clean_training_data(train_data, anomaly_windows)

    # Scale training data
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(cleaned_training_data)
    #print(scaled_train_data)
    scaled_test_data = scaler.transform(test_data)
    train_loader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data,batch_size=batch_size, shuffle=False, drop_last=False)
    iteration_counter = 0
    
    # print(len(scaled_test_data))
    # print(len(scaled_train_data))
    # print(reconstruction_error_full)
    
    mem_rec_full = []
    for epoch in range(epochs):
        item_num = 0
        mem_rec_full = []
        reconstruction_error_full =[0]*len(scaled_test_data)
        print(epoch)
        train_batch = iter(train_loader)
        #print(train_loader)
        for data in train_batch:
            iteration_counter = iteration_counter+1
            #data = torch.from_numpy(data)
            #print(data)
            data = data.to(torch.float)
            data = data.view(-1, 2410).to(device)
            targets = data.to(device)
            
            model.train()
            #print(data.dtype)
            mem_rec = model(data)
            #print(mem_rec)
            # loss_val = torch.zeros((1), dtype=dtype, device=device)
            # for step in range(num_steps):
            #     loss_val += loss(mem_rec[step], targets)
            #print(data)
            #a = mem_rec.detach().numpy()
            #a = mem_rec.numpy()
            #data = data.numpy()

            # reconstruction_error_piece = torch.mean(
            #     torch.pow(data- mem_rec, 2), axis=1
            # )

            # print(reconstruction_error_piece)
            #print(a)
            #print(mem_rec, 1)
            # for i in mem_rec:
            #     print(i)
            #     i = i.detach()
            #     print(i , 'detached')
            #     mem_rec_full.append(i)
                #print(mem_rec_full)
            #print(mem_rec_full)
            
        #print(mem_rec_full)
        # print(scaled_train_data)
            scaled_train_data = torch.tensor(scaled_train_data)
            #mem_rec_detached = torch.FloatTensor(mem_rec_detached)
            # print(scaled_train_data)
            # print(mem_rec_detached)
            #mem_rec_full = torch.tensor(mem_rec_full)
            #print(mem_rec,2)
            #print(data,3)
            loss = criterion(mem_rec, data)
            optimizer.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
            #print(reconstruction_error_full)
            # print(loss_val)
            # targets = torch.tensor([2409])

            # test_data = torch.tensor(scaled_test_data)
            # for step in range(num_steps):
            #     loss_val += loss(mem_rec[step], test_data[step])

            # optimizer.zero_grad()
            # #loss_val.backward()
            # optimizer.step()
            # loss_hist.append(loss_val.item())



        # Compute anomaly likelihoods

        #SOMETHING HERE FOR EXECUTION?
    
    return model
    #return pd.Series(is_anomalies, train_data.index), likelihoods, reconstruction_error_full
        #print(likelihoods)


            # with torch.no_grad():
            #     model.eval()
            #     test_data = torch.tensor(test_data)
            #     test_data = test_data.to(float)
            #     test_data = test_data.to(device)
            #     test_targets = test_targets.to(device)

            #     test_spk, test_mem = model(test_data.view(batch_size, -1))

            #     test_loss = torch.zeros((1), dytpe=dtype, device=device)

            #     for step in range(num_steps):
            #         test_loss += loss(test_mem[step], test_targets)
            #     test_loss_hist.append(test_loss.item())

            # if counter % 50 == 0:
            #     train_printer()
            # counter += 1
            # iteration_counter += 1
# def compute_anomaly_likelihood(historical_recon_error, anomaly_long_window_length, anomaly_short_window_length):
#     """
#     Computes the anomaly likelihood based on reconstruction error and statistical analysis.

#     Parameters:
#     - historical_recon_error: Array, historical reconstruction error values.
#     - anomaly_long_window_length: Integer, length of the long window for analysis.
#     - anomaly_short_window_length: Integer, length of the short window for analysis.

#     Returns:
#     - likelihood: Float, computed anomaly likelihood.
#     """
#     if len(historical_recon_error) <= anomaly_long_window_length:
#         return 0.5

#     wide_data_window = historical_recon_error[-anomaly_long_window_length:]
#     narrow_data_window = historical_recon_error[-anomaly_short_window_length:]

#     mean_of_wide_data_window = np.mean(wide_data_window)
#     stdev_of_wide_data_window = np.std(wide_data_window)

#     # Fixed epsilon to avoid division by zero
#     epsilon = 1e-10
#     likelihood = 0.5 + 0.5 * (
#         1 - norm.sf(
#             (np.mean(narrow_data_window) - mean_of_wide_data_window) / (stdev_of_wide_data_window + epsilon)
#         )
#     )

#     return likelihood

def print_batch_accuracy(data, targets, train=False):
    output, _ = model(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(epoch, iter_counter, loss_hist, counter, test_loss_hist, data, targets, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

def evaluate_performance(y_true, y_pred):
    """
    Evaluates the performance of the anomaly detection using various classification metrics.

    Parameters:
    - y_true: Array, true labels.
    - y_pred: Array, predicted labels.

    Returns:
    - precision: Float, precision of the model.
    - recall: Float, recall of the model.
    - f1: Float, F1-score of the model.
    - accuracy: Float, accuracy of the model.
    - conf_matrix: Array, confusion matrix.
    - mcc: Float, Matthews correlation coefficient.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return precision, recall, f1, accuracy, conf_matrix, mcc

#def testing(test_data, anomaly_windows_test, is_anomalies, likelihoods, reconstruction_error):
def testing(test_data, anomaly_windows_test, model):

        y_true = [1 if ((anomaly_windows_test['anomaly_window_start'] <= t) & (t <= anomaly_windows_test['anomaly_window_end'])).any() else 0 for t in test_data.index]
        scaler = MinMaxScaler()
        #test_data = clean_training_data(test_data, anomaly_windows_test)
        scaled_test_data = scaler.fit_transform(test_data)
        #print(y_true)
        # for i in is_anomalies:
        #     if i == 1:
        #         print(i)

        
        reconstruction_error_full = []
        test_loader = DataLoader(scaled_test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        with torch.no_grad():
            model.eval()
            for data in iter(test_loader):
                data = data.to(torch.float)
                data = data.to(device)
                tha = model(data)
                reconstruction_error_piece = torch.mean(
                    torch.pow(data- tha, 2), axis=1
                )
                for i in reconstruction_error_piece:
                    #print(i)
                    i = i.detach()
                    #if i > 0.245 or i < 0.24:
                        #print(i)
                    reconstruction_error_full.append(i)
                #print(tha)
        
        #print(reconstruction_error_full)
        likelihoods = []
        print('Computing Likelihood')
        for i in range(len(reconstruction_error_full)):
            if(i%10000 == 0):
                print(i)
            #print(reconstruction_error_full[:i+1])
            likelihood = compute_anomaly_likelihood(
                reconstruction_error_full[:i + 1], long_window, short_window
            )
            likelihoods.append(likelihood)

        #print(likelihoods)
        #print(anomaly_threshold)
        reconstruction_error = reconstruction_error_full
        likelihoods = np.array(likelihoods)
        is_anomalies = (likelihoods > anomaly_threshold).astype(int)

        #TEST DATA SHOULD BE SCALED!
        #WE NEED X_TEST_PREDICTIONS FROM OG
        #HOW DO I DO THAT WITH PYTORCH

        #print(y_true)
        # print(is_anomalies)
        # Create results DataFrame for evaluation
        #print(y_true)
        #print(is_anomalies)
        print(len(scaled_test_data))
        print(len(y_true))
        print(is_anomalies.size)
        print(likelihoods.size)
        print(len(reconstruction_error))
        print(is_anomalies)
        results_df = pd.DataFrame({
            '5XX_count': test_data.sum(axis=1),  # Adjust as needed for your data
            'true_anomaly': y_true,  # This is what the function expects
            'predicted_anomaly': is_anomalies,  # The output of the model
            'anomaly_likelihood': likelihoods,
            'reconstruction_error': reconstruction_error
        })
        #print(results_df)
        print("SAMPLE RESULT DF: ", results_df.head())
        #print(y_true)
        # Evaluate performance based on ground truth

        precision, recall, f1, accuracy, conf_matrix, mcc = evaluate_performance(y_true, is_anomalies)

        log.info(f"Experiment results - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, ConfusionMatrix: {conf_matrix}")

        #Calculate the weighted NAB score and normalized NAB score
        raw_nab_score, normalized_nab_score, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
            results_df, anomaly_windows_test, nab_scoring_profile, true_col='true_anomaly', pred_col='predicted_anomaly'
        )

        print(raw_nab_score)
        print(normalized_nab_score)
        print(false_positive_count)
        print(false_negative_count)
        print(detection_counters)
        # Return all values needed for grid_search
        #return precision, recall, f1, accuracy, conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score, normalized_nab_score, detection_counters
    # total = 0
    # correct = 0
    # test_loader = DataLoader(test_data,batch_size=batch_size, shuffle=False, drop_last=True)
    # with torch.no_grad():
    #     model.eval()
    #     for data in iter(test_loader):
    #         print(data)
    #         data = torch.from_numpy(data)
    #         data = data.to(device)
    #         #targets = targets.to(device)

    #         test_spk, _ = model(data.view(data.size(0), -1))

    #         _, predicted = test_spk.sum(dim=0).max(1)
    #         #total += targets.size(0)
    #         #correct += (predicted == targets).sum().item()

# def final_thing():
#     log.info(f"Raw NAB Score: {raw_nab_score}, Normalized NAB Score: {normalized_nab_score}")
#     print("++++++detection_counters_nab: ", detection_counters_nab)

#     # Collect results
#     detection_counters = {
#         'tp': conf_matrix[1, 1],
#         'tn': conf_matrix[0, 0],
#         'fp': conf_matrix[0, 1],
#         'fn': conf_matrix[1, 0],
#         # 'issue_detected': anomaly_windows[anomaly_windows['anomaly_source'] == 1].shape[0],
#         # 'im_detected': anomaly_windows[anomaly_windows['anomaly_source'] == 2].shape[0],
#         # 'TestLog_detected': anomaly_windows[anomaly_windows['anomaly_source'] == 3].shape[0],
#     }

#     # Use detection counters from run_experiment or add it separately if needed
#     result = {
#         'Threshold': anomaly_threshold,
#         'Long Window': long_window,
#         'Short Window': short_window,
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'Accuracy': accuracy,
#         'MCC': mcc,
#         # 'TP': detection_counters['tp'],
#         'TP': detection_counters_nab['tp'],
#         'TP_Org': detection_counters['tp'],
#         'FN_Org': detection_counters['fn'],
#         'TN': detection_counters['tn'],
#         'FP': detection_counters['fp'],
#         'FN': detection_counters_nab['fn'], # It should window based; not point based
#         'Raw NAB Score': raw_nab_score,
#         'Normalized NAB Score': normalized_nab_score,
#         'Problem Detected': detection_counters_nab['issue_detected'],
#         'Slack Detected': detection_counters_nab['im_detected'],
#         'Incident Detected': detection_counters_nab['TestLog_detected']
#     }

#     all_results.append(result)

#     # Check if this is the best score so far based on normalized NAB score
#     if normalized_nab_score > best_score:
#         best_score = normalized_nab_score
#         best_params = (anomaly_threshold, long_window, short_window)
#         best_results = result

#     # Ensure the directory exists
#     dir_to_save_to = os.path.dirname(result_directory)
#     if dir_to_save_to and not os.path.exists(dir_to_save_to):
#         os.makedirs(dir_to_save_to)

#     results_df_all = pd.DataFrame(all_results)
#     # Define a file path variable
#     filename_prefix = 'weighted_' if use_weights else 'unweighted_'
#     csv_filepath = os.path.join(result_directory, f'{filename_prefix}_{model}_results.csv')
#     results_df_all.to_csv(csv_filepath, index=False)

#     # Save results to CSV
#     unweighted_csv_filepath = os.path.join(result_directory, (filename_prefix + '_' + model + '_anomaly_detection_results.csv'))
#     results_df.to_csv(unweighted_csv_filepath, index=True)

#     log.info(f"Best Params: {best_params}")
#     log.info(f"Best Raw NAB Score: {best_results['Raw NAB Score']}")
#     log.info(f"Best Normalized NAB Score: {best_score}")


#     # Plot the result
#     plot_results(
#         results_df=results_df,
#         is_anomalies=is_anomalies,
#         # anomaly_windows=anomaly_windows,
#         anomaly_windows=anomaly_windows_test,
#         result_directory=result_directory,
#         file_name_prefix='unweighted_',
#         raw_nab_score=raw_nab_score,  # Pass raw NAB score here
#         normalized_nab_score=normalized_nab_score,  # Pass normalized NAB score here
#         model=model
#     )

#     return best_params, best_score, best_results, csv_filepath


def main():
    """
    Main function to coordinate anomaly detection using autoencoders.

    Steps:
    1. Load and preprocess the data.
    2. Extract training and testing windows.
    3. Prepare ground truth anomaly windows for evaluation.
    4. Perform grid search to optimize anomaly detection parameters.

    Parameters:
    - cfg: DictConfig, configuration object containing paths, parameters, and settings.
    """
    log.info("Starting main function")

    # Load and prepare data
    #df = pd.read_parquet("C:\schoolstuff\\thesis2fold\PQ_data\\14062900 (1)\pivoted_data_all.parquet")
    log.info("loading data")
    df_prodlive, gt_utc_df = load_and_prepare_data()
    df_prodlive = add_time_features(df_prodlive)

    # Extract experiment parameters
    start_date = pd.Timestamp('2024-01-26')
    train_end_date = pd.Timestamp('2024-02-29')
    test_start_date = pd.Timestamp('2024-03-01')
    end_date = pd.Timestamp('2024-05-31') + timedelta(days=1)
    result_directory = "./results/model_experiments/"
    model_save_directory = "./trained_models/"





    # Filter data for training and testing
    working_data = df_prodlive.loc[start_date:end_date]
    train_data = working_data.loc[start_date:train_end_date]
    dir_to_save_to = os.path.dirname("C:\schoolstuff\\thesis\data\\5xx_feature_dir")
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    # Save to CSV
    train_data.to_csv("C:\schoolstuff\\thesis\data\\5xx_feature_dir\\train.csv", index=True)
    test_data = working_data.loc[test_start_date:end_date]
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)
    test_data.to_csv("C:\schoolstuff\\thesis\data\\5xx_feature_dir\\test.csv", index = True)
    # Prepare anomaly windows
    anomaly_windows, anomaly_windows_test = filter_anomaly_windows(gt_utc_df, start_date, end_date, test_start_date)
    log.info(
        f"Anomaly windows prepared. Training windows: {len(anomaly_windows)}, Testing windows: {len(anomaly_windows_test)}")


    log.info(f'Starting Training')
    model = training(train_data, anomaly_windows, test_data)
    log.info(f'Starting Testing')
    testing(test_data, anomaly_windows_test, model)
if __name__ == "__main__":
    main()

