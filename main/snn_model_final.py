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
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Net(nn.Module):
    def __init__(self):
        super().__init__()



        self.encoder = nn.Sequential(

            nn.Linear(2410, 128),
            snn.Leaky(beta=0.95, threshold=1, init_hidden=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            snn.Leaky(beta=0.95, threshold=1,  init_hidden=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 14),
            snn.Leaky(beta=0.95, threshold=1, output=True, init_hidden=True),
            
        )


        self.decoder = nn.Sequential(
            nn.Linear(14, 64),
            snn.Leaky(beta=0.95, threshold=1, init_hidden=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            snn.Leaky(beta=0.95, threshold=1, init_hidden=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 2410),
            snn.Leaky(beta=0.95, threshold=1,output=True, init_hidden=True),

        )
    

    def forward(self, x):

        encoded, mem_x = self.encoder(x)
        decoded, mem_dec = self.decoder(encoded)
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        return decoded

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
#criterion = nn.BCELoss()
criterion = nn.MSELoss()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
dtype = torch.float64
batch_size = 32
train_set = pd.read_csv("..\\thesis2fold\\data\\5xx_feature_dir\\train.csv")

anomaly_threshold = 0.9996
long_window = 30
short_window = 2
nab_scoring_profile = "reward_fn"

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
    #scaled_test_data = scaler.fit_transform(test_data)
    test_loader = DataLoader(scaled_test_data,batch_size=batch_size, shuffle=False, drop_last=False)
    iteration_counter = 0
    
    model.train()
    for epoch in range(epochs):
        print(epoch)
        train_batch = iter(train_loader)
        for data in train_batch:
            iteration_counter = iteration_counter+1
            data = data.to(torch.float)
            data = data.view(-1, 2410).to(device)
            targets = data.to(device)
            
            
            mem_rec = model(data)

            scaled_train_data = torch.tensor(scaled_train_data)
            loss = criterion(mem_rec, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scaler = MinMaxScaler()
           
    
    return model


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
                # print(tha)
                # print(data)
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

        reconstruction_error = reconstruction_error_full
        likelihoods = np.array(likelihoods)
        is_anomalies = (likelihoods > anomaly_threshold).astype(int)

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
        print(conf_matrix)
        log.info(f"Experiment results - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, ConfusionMatrix: {conf_matrix}")

        return results_df, is_anomalies, raw_nab_score, normalized_nab_score
        


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
    #df = pd.read_parquet("..\\thesis2fold\PQ_data\\14062900 (1)\pivoted_data_all.parquet")
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
    dir_to_save_to = os.path.dirname("..\\5xx_feature_dir")
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    # Save to CSV
    train_data.to_csv("..\\thesis2fold\\data\\5xx_feature_dir\\train.csv", index=True)
    test_data = working_data.loc[test_start_date:end_date]
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)
    test_data.to_csv("..\\thesis2fold\\data\\5xx_feature_dir\\test.csv", index = True)
    # Prepare anomaly windows
    anomaly_windows, anomaly_windows_test = filter_anomaly_windows(gt_utc_df, start_date, end_date, test_start_date)
    log.info(
        f"Anomaly windows prepared. Training windows: {len(anomaly_windows)}, Testing windows: {len(anomaly_windows_test)}")


    log.info(f'Starting Training')
    start = time.time()
    model = training(train_data, anomaly_windows, test_data)
    end = time.time()
    
    log.info(f'Starting Testing')
    test_start = time.time()
    results_df, is_anomalies, raw_nab_score, normalized_nab_score = testing(test_data, anomaly_windows_test, model)
    test_end = time.time()
    plot_results(results_df, is_anomalies, anomaly_windows, result_directory, raw_nab_score=raw_nab_score, normalized_nab_score=normalized_nab_score, model='SNN')
    print("Training Time ", end - start)
    print("Test Time ", test_end-test_start)
if __name__ == "__main__":
    main()

