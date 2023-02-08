# SlowFastSeparation

Codes for submitted paper in KDD'23: "Learning Slow and Fast System Dynamics via Automatic Separation of Time Scales"



**The refactored version coming soon!**



# Requirements
- Python 3.10

- PyTorch==1.12

- scikit-learn==1.1.2

- Numpy

- Scipy

- Matplotlib

# Usage

Train and test model in 1S2F system:

```shell
cd 1S2F

python main.py # for our model

python lstm.py # for LSTM

python tcn.py # for TCN

python neural_ode.py # for Neural ODE
```

Train and test model in 2S2F system:

```shell
cd 2S2F

python main.py # for our model

python lstm.py # for LSTM

python tcn.py # for TCN

python neural_ode.py # for Neural ODE
```