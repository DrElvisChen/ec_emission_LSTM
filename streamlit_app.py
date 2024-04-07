from operator import index
import streamlit as st

from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objs as go
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://impacts.cloud/assets/img/logo/logo.png")
    st.title("CRIS_AutoML")
    choice = st.radio("Navigation", ["Upload","Detail in each machine","Modelling", "Time Series Forecasting"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
if choice == "Detail in each machine":
    st.subheader("Distributions of each columns")
    options = (
    "static frequency converter", "circulating water pump", "closed circulating water pump", "cooling tower fan",
    "condensate pump", "high-pressure boiler feed pump", "Gas turbine operating status", "Carbon dioxide emissions")
    sel_cols = st.selectbox("select columns", options, 1)
    st.write(sel_cols)

    # Define a dictionary with image paths
    images = {
        "static frequency converter": "static frequency converter.jpg",
        "circulating water pump": "circulating water pump.jpg",
        "closed circulating water pump": "closed circulating water pump.jpg",
        "cooling tower fan": "cooling tower fan.jpg",
        "condensate pump": "condensate pump.jpg",
        "high-pressure boiler feed pump": "high-pressure boiler feed pump.jpg",
        "Gas turbine operating status": "Gas turbine operating status.jpg",
        "Carbon dioxide emissions": "Carbon dioxide emissions.jpg"
    }

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the image in the first column
    col1.image(images[sel_cols])

    # Display descriptive statistics of the selected column in the second column
    col2.table(df[sel_cols].describe())

    # Change histogram to line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[sel_cols], mode='lines'))
    st.plotly_chart(fig)

if choice == "Time Series Forecasting":
    st.subheader("Time Series Forecasting<LSTM, GRU and RNN>")
    st.empty()
    my_bar = st.progress(0)
    st.image("C:\\Users\\ElvisChen\\PycharmProjects\\LSTM\\model.jpg")

    # Prepare the data
    features = df.drop('Carbon dioxide emissions', axis=1).values
    target = df['Carbon dioxide emissions'].values

    # Convert to PyTorch tensors
    features = torch.Tensor(features)
    target = torch.Tensor(target)

    # Create a TensorDataset and a DataLoader
    dataset = TensorDataset(features, target)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    # Define the LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


    # Define the GRU model
    # Define the GRU model
    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GRU, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, hidden = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out


    # Define the RNN model
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)  # Pass h0 directly, not as a tuple
            out = self.fc(out[:, -1, :])
            return out


    # Instantiate the models
    lstm_model = LSTM(input_size=features.shape[1], hidden_size=50, num_layers=1, output_size=1)
    gru_model = GRU(input_size=features.shape[1], hidden_size=50, num_layers=1, output_size=1)
    rnn_model = RNN(input_size=features.shape[1], hidden_size=50, num_layers=1, output_size=1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)


    # Train the models
    # Train the models
    def train_model(model, dataloader, num_epochs, criterion, optimizer):
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs = inputs.unsqueeze(1)  # Add an extra dimension for time steps
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update the progress bar
            progress = (epoch + 1) / num_epochs
            my_bar.progress(progress)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate the models
    def evaluate_model(model, features):
        with torch.no_grad():
            inputs = features.unsqueeze(1)
            outputs = model(inputs)
        return outputs.squeeze().detach().numpy()


    # Train and evaluate the models
    my_bar = st.progress(0)
    train_model(lstm_model, dataloader, 500, criterion, lstm_optimizer)
    lstm_predictions = evaluate_model(lstm_model, features)

    train_model(gru_model, dataloader, 300, criterion, gru_optimizer)
    gru_predictions = evaluate_model(gru_model, features)
    train_model(rnn_model, dataloader, 300, criterion, rnn_optimizer)
    rnn_predictions = evaluate_model(rnn_model, features)
    # Save the models
    torch.save(lstm_model.state_dict(), 'LSTM.pth')
    torch.save(gru_model.state_dict(), 'GRU.pth')
    torch.save(rnn_model.state_dict(), 'RNN.pth')

    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return mae, mape, r2


    # Display metrics in a table
    # Convert target to numpy array
    target_np = target.numpy()

    # Calculate metrics
    lstm_metrics = calculate_metrics(target_np, lstm_predictions)
    gru_metrics = calculate_metrics(target_np, gru_predictions)
    rnn_metrics = calculate_metrics(target_np, rnn_predictions)

    metrics_df = pd.DataFrame([lstm_metrics, gru_metrics, rnn_metrics],
                              columns=['MAE', 'MAPE', 'R2'],
                              index=['LSTM', 'GRU', 'RNN'])

    st.write(metrics_df)
    # Create a DataFrame for the data
    # Create individual DataFrames for each model
    data_lstm = pd.DataFrame({
        'Real': target_np,
        'LSTM Predictions': lstm_predictions
    })

    data_gru = pd.DataFrame({
        'Real': target_np,
        'GRU Predictions': gru_predictions
    })

    data_rnn = pd.DataFrame({
        'Real': target_np,
        'RNN Predictions': rnn_predictions
    })

    # Create the lineplot for LSTM
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data_lstm, palette=["blue", "red"])
    plt.title('Real vs LSTM Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    st.pyplot(plt.gcf())  # Display the plot in Streamlit

    # Create the lineplot for GRU
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data_gru, palette=["blue", "green"])
    plt.title('Real vs GRU Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    st.pyplot(plt.gcf())  # Display the plot in Streamlit

    # Create the lineplot for RNN
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data_rnn, palette=["blue", "purple"])
    plt.title('Real vs RNN Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    st.pyplot(plt.gcf())  # Display the plot in Streamlit


