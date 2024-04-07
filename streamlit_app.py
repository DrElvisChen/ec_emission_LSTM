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
    choice = st.radio("Navigation", ["Upload","Detail in each machine"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

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

