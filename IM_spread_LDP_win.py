from collections import defaultdict
import re
from utils import *


def IM_spread(dataset_name, file_name, seed_size, time_index):
    if dataset_name == 'EmailDept1':
        data_path = f'./real_data/EmailDept1_LDP/{dataset_name}_{time_index}.txt'
    elif dataset_name == 'FbForum':
        data_path = f'./real_data/Forum_LDP/{dataset_name}_{time_index}.txt'
    else:
        data_path = f'./real_data/Tech_LDP/{dataset_name}_{time_index}.txt'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The original data file {data_path} does not exist.")

    # Get the seed nodes and calculate the influence spread
    S = find_seed(file_name, seed_size=seed_size)
    influence_spread = cal_spread(data_path, S_all=S, seed_size=seed_size)
    return influence_spread

if __name__ == '__main__':
    dataset_name = 'FbForum'
    #w = [3, 6, 9, 12, 15]
    w = [6]
    seed_size = 20
    base_dir = './result/SynGraph_Save_vary_w'

    spread_all = [] # Save the average result for each window
    for wi in range(len(w)):
        window = w[wi]

        time_spreads = defaultdict(list)

        for root, dirs, files in os.walk(os.path.join(base_dir, dataset_name, f'win{window:.1f}')):
            for file in files:
                if file.startswith('WGT_') and file.endswith('.txt'):
                    match = re.match(r'WGT_(.+?)_(\d+\.\d+)_(\d+)_(\d+)\.txt', file)
                    if not match:
                        print(f"Skip malformed files: {file}")
                        continue

                    # Extract parameters
                    file_dataset = match.group(1)
                    file_window = float(match.group(2))
                    file_time = int(match.group(3))
                    file_exper = int(match.group(4))

                    # Filter matching conditions
                    if file_dataset == dataset_name and abs(file_window - window) < 1e-6:
                        file_path = os.path.join(root, file)
                        try:
                            spread = IM_spread(dataset_name, file_path, seed_size, file_time)
                            time_spreads[file_time].append(spread)
                        except FileNotFoundError as e:
                            print(f"Calculation failed: {e}")
                            continue

        avg_spread = []

        for time_index in sorted(time_spreads.keys()):
            spreads = time_spreads[time_index]
            current_avg = sum(spreads) / len(spreads)
            avg_spread.append(current_avg)
            print(f"Average influence spread of time index {time_index}: {current_avg:.2f}")

        spread_all.append(sum(avg_spread) / len(avg_spread))

    print('dataset:', base_dir + '/' + dataset_name)
    print('window=', w)
    print('spread_all=', spread_all)
