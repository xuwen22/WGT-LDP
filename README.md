# WGT-LDP
This is the code repository for our paper "Continuous Publication of Weighted Graphs with Local Differential Privacy". 
## Requirements
To run the experiments in this repository, you need *numpy*, *networkx*, *scikit-learn*. You can install all the packages is through conda and pip:
```python
pip install numpy
pip install networkx
pip install scikit-learn
```
## File Contents
- `data` is the folder with all datasets.
  - `EmailDept1_LDP` is the Email-Eu dataset in our paper, which contains dynamic communications between 319 nodes over 173 time steps.
  - `Forum_LDP` is the Forum dataset in our paper, which contains dynamic interaction records between 899 students over 24 time steps.
  - `Tech_LDP` is the Tech-AS dataset in our paper, which contains dynamic connections between 5000 autonomous systems over 24 time steps.
- `result` is the folder for storing synthesized weighted graph (.txt format), including two examples.
    -`SynGraph_Save` stores the average results of ten runs on the Forum dataset when the privacy budget is 1.
    -`SynGraph_Save_vary_w` stores the average results of ten runs on the Forum dataset when the di!erent sliding window size is 6.
- `main.py` is the file used to run the WGT-LDP framework with different privacy budgets.
- `main_vary_w.py` is the file used to run the WGT-LDP framework with different sliding windows.
- `main_event.py` is the file used to run the WGT-LDP framework with event-level privacy.
- `utils.py` includes some functions that are needed for other files.
- `IM_spread_LDP_eps.py` is used to obtain the results of influence maximization under different privacy budgets.
- `IM_spread_LDP_win.py` is used to obtain the results of influence maximization under different sliding windows.
## Running
you can run experiments with `python {file name}.py`. For example,
```python
python main.py
```
