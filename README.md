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
- `result` is the folder for storing synthetic weighted graph (.txt format), including two examples.
  - `SynGraph_Save` stores the average results of ten runs on the Forum dataset when the privacy budget is 1.
  - `SynGraph_Save_vary_w` stores the average results of ten runs on the Forum dataset when the sliding window size is 6.
- `main.py` is the file used to run the WGT-LDP framework with different privacy budgets.
- `main_vary_w.py` is the file used to run the WGT-LDP framework with different sliding windows.
- `main_event.py` is the file used to run the WGT-LDP framework with event-level privacy.
- `utils.py` includes some functions that are needed for other files.
- `IM_spread_LDP_eps.py` is used to obtain the results of influence maximization under different privacy budgets.
- `IM_spread_LDP_win.py` is used to obtain the results of influence maximization under different sliding windows.
## Running
### I. main Experiments
you can run main experiments with `python {file name}.py`.
```python
###### Example 1 ######
python main.py

###### Example 2 ######
python main_vary_w.py

###### Example 3 ######
python main_event.py
```
**Default dataset**: Forum (pre-configured).

**To switch datasets** (e.g. to Tech-AS):
- Comment out Forum block in your target main_*.py file:
```python
# dataset: Forum
# data_path = "./data/Forum_LDP/FbForum"
# node_num = 899
# snapshot_num = 24
# max_h = 168
```
- Uncomment target dataset block:
```python
# dataset: Tech-AS
data_path = "./data/Tech_LDP/tech"
node_num = 5000
snapshot_num = 24
max_h = 24
```

### II. Influence Maximization Experiments
you can run influence maximization experiments with `python {file name}.py`.
```python
###### Example 1 ######
python IM_spread_LDP_eps.py

###### Example 2 ######
python IM_spread_LDP_win.py
```

**Preparation** (required before running IM experiments):
1. **Run a main experiment with graph saving enabled**:
- In the main experiment file (e.g., `main.py`), uncomment:
  - Target `dataset_name` assignment.
  - `utils.save_graph_with_params(...)` call.
```python
###### Example ######
# Before:
# dataset_name = "tech"
# utils.save_graph_with_params(dataset_name, epsilon, ...)
# After:
dataset_name = "tech"
utils.save_graph_with_params(dataset_name, epsilon, ...)
```
> **Note**: Run main experiment after this change to generate synthetic graph data (.txt format).
2. **Configure the IM experiment file**:
- In `IM_spread_LDP_eps.py` or `IM_spread_LDP_win.py`, set `dataset_name` to match:
```python
# For Tech-AS:
dataset_name = 'tech'
```

### Important Notes
- **Dataset Consistency**: The dataset name must be identical in the main experiment (when saving) and the IM experiment.
- **File-Specific Configuration**: Each main experiment file (main.py, main_vary_w.py, main_event.py) has its own configuration block. If you switch datasets, you must update the configuration in the specific main file you are running.
- **Graph Saving**: Only uncomment the saving code when you intend to generate data for IM experiments (it may slow down the main experiment).
