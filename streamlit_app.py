from operator import index
import streamlit as st
from pycaret.regression import setup, compare_models, pull, save_model, load_model
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
    choice = st.radio("Navigation", ["Upload","Detail in each machine","Heatmap","Modelling", "Time Series Forecasting","Carbon dioxide emission predictor","Carbon dioxide emission predictor-upload"])
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
if choice == "Heatmap":
    st.title("Heatmap")
    plt.figure(figsize=(16, 9))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=2)
    st.pyplot(plt)
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
        loss_values = []  # List to store loss values
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

            # Store the loss value
            loss_values.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        return loss_values  # Return the list of loss values

    # Evaluate the models
    def evaluate_model(model, features):
        with torch.no_grad():
            inputs = features.unsqueeze(1)
            outputs = model(inputs)
        return outputs.squeeze().detach().numpy()


    # Train and evaluate the models
    my_bar = st.progress(0)
    # Train the models and get the loss values
    loss_values_lstm = train_model(lstm_model, dataloader, 500, criterion, lstm_optimizer)

    loss_values_gru = train_model(gru_model, dataloader, 300, criterion, gru_optimizer)

    loss_values_rnn = train_model(rnn_model, dataloader, 300, criterion, rnn_optimizer)

    # Plot the loss values
    plt.figure(figsize=(12, 6))
    plt.plot(loss_values_lstm, label='LSTM')
    plt.plot(loss_values_gru, label='GRU')
    plt.plot(loss_values_rnn, label='RNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss values per epoch for each model')
    st.pyplot(plt.gcf())  # Display the plot in Streamlit
    lstm_predictions = evaluate_model(lstm_model, features)
    gru_predictions = evaluate_model(gru_model, features)
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
if choice == "Carbon dioxide emission predictor":
    # Define the LSTM, GRU, RNN models as before
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
    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GRU, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
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
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out


    def load_model(model_selection):
        input_size = 7  # Assuming 7 input features based on your inputs
        hidden_size = 50  # Example hidden size, adjust based on your model design
        num_layers = 1  # Adjust this to match the number of layers in your saved model
        output_size = 1  # Assuming a single output value

        if model_selection == "LSTM":
            model = LSTM(input_size, hidden_size, num_layers, output_size)
        elif model_selection == "RNN":
            model = RNN(input_size, hidden_size, num_layers, output_size)
        elif model_selection == "GRU":
            model = GRU(input_size, hidden_size, num_layers, output_size)
        else:
            raise ValueError("Invalid model selection")

        # Get the current working directory
        current_dir = os.getcwd()

        # Construct the model file path relative to the current working directory
        model_path = os.path.join(current_dir, f"{model_selection}.pth")  # Adjusted to .pth

        # Load the model state dictionary
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        return model


    def main():
        st.title("Carbon dioxide emissions Predictor")
        html_temp = """
        <div style="background:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Carbon dioxide emissions Predictor</h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Input fields
        static_frequency_converter = st.number_input("Static Frequency Converter", min_value=0, max_value=200,
                                                     value=100)
        circulating_water_pump = st.number_input("Circulating Water Pump", min_value=0, max_value=200, value=100)
        condensate_pump = st.number_input("Condensate Pump", min_value=0, max_value=200, value=100)
        closed_circulating_water_pump = st.number_input("Closed Circulating Water Pump", min_value=0, max_value=200,
                                                        value=100)
        cooling_tower_fan = st.number_input("Cooling Tower Fan", min_value=0, max_value=200, value=100)
        high_pressure_boiler_feed_pump = st.number_input("High Pressure Boiler Feed Pump", min_value=0, max_value=200,
                                                         value=100)
        gas_turbine_operating_status = st.number_input("Gas Turbine Operating Status", min_value=-1, max_value=1,
                                                       value=0)

        # Collecting inputs into an array
        input_data = np.array([
            static_frequency_converter,
            circulating_water_pump,
            condensate_pump,
            closed_circulating_water_pump,
            cooling_tower_fan,
            high_pressure_boiler_feed_pump,
            gas_turbine_operating_status
        ]).reshape(1, -1)  # Reshaping to match the expected input dimensions

        model_selection = st.selectbox("Model", ["LSTM", "RNN", "GRU"])

        if st.button("Predict"):
            model = load_model(model_selection)
            input_tensor = torch.from_numpy(input_data.astype(np.float32))
            input_tensor = input_tensor.unsqueeze(0)  # Add sequence dimension

            # Make a prediction
            with torch.no_grad():
                prediction = model(input_tensor)
                st.write(f"Predicted Carbon dioxide emissions: {prediction.item()}")


    if __name__ == '__main__':
        main()
if choice == "Carbon dioxide emission predictor-upload":

    # Define the LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            out, _ = self.lstm(x.unsqueeze(0), (h0.unsqueeze(0), c0.unsqueeze(0)))
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
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            out, _ = self.gru(x.unsqueeze(0), h0.unsqueeze(0))
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
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            out, _ = self.rnn(x.unsqueeze(0), h0.unsqueeze(0))  # Pass h0 directly, not as a tuple
            out = self.fc(out[:, -1, :])
            return out

    def load_model(model_selection):
        input_size = 7  # Assuming 7 input features based on your inputs
        hidden_size = 50  # Example hidden size, adjust based on your model design
        num_layers = 1  # Adjust this to match the number of layers in your saved model
        output_size = 1  # Assuming a single output value

        if model_selection == "LSTM":
            model = LSTM(input_size, hidden_size, num_layers, output_size)
        elif model_selection == "GRU":
            model = GRU(input_size, hidden_size, num_layers, output_size)
        elif model_selection == "RNN":
            model = RNN(input_size, hidden_size, num_layers, output_size)
        else:
            raise ValueError("Invalid model selection")

        model_path = os.path.join(r"C:\Users\ElvisChen\PycharmProjects\LSTM", f"{model_selection}.pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model


    def make_predictions(model, input_data):
        input_tensor = torch.from_numpy(input_data.astype(np.float32))
        input_tensor = input_tensor.unsqueeze(0)  # Adding a batch dimension
        with torch.no_grad():
            prediction = model(input_tensor)
        return prediction.numpy().flatten()


    def main():
        st.title("Carbon Dioxide Emissions Predictor")

        st.markdown("""
        <div style="background-color:#025246;padding:10px">
        <h2 style="color:white;text-align:center;">Predict CO2 emissions using LSTM, GRU, or RNN models</h2>
        </div>
        """, unsafe_allow_html=True)

        model_selection = st.sidebar.selectbox("Choose a model for prediction", ["LSTM", "GRU", "RNN"])
        choice = st.sidebar.radio("Choose input method", [ "Upload CSV"])


        if choice == "Upload CSV":
            st.subheader("CSV data input")
            uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)

                if data.shape[1] != 7:
                    st.error("CSV file must have 7 columns.")
                else:
                    model = load_model(model_selection)
                    input_data = data.values.astype(np.float32)
                    predictions = np.apply_along_axis(lambda row: make_predictions(model, row), 1, input_data)

                    # Add predictions to the DataFrame
                    data['Predicted CO2 Emissions'] = predictions
                    st.write(data)


    if __name__ == "__main__":
        main()