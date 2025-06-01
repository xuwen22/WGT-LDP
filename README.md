# WGT-LDP
Implementation of WGT-LDP
## Requirements
To run the experiments in this repository, you need *numpy*, *networkx*, *scikit-learn*.
## File Contents
- `data` is the folder with all datasets.
- `result` is the folder for storing synthesized weighted graph (.txt format), including two examples.
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
