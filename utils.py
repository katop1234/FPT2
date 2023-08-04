import os
import pickle
import pyarrow.feather as feather
import torch.distributed as dist
import torch

serialized_dir = "/shared/katop1234/FPT2/serialized/"

# Efficient serialization for large objects
def write_feather(df, path):
    feather.write_feather(df, path)

# Efficient deserialization for large objects
def read_feather(path):
    return feather.read_feather(path)

# Write objects to serialized/
def write(obj, obj_name):
    if not os.path.exists(serialized_dir):
        os.mkdir(serialized_dir)
    if obj_name == "SNPdata.ser":
        return write_feather(obj, serialized_dir + obj_name)
    dir_path = serialized_dir
    with open(dir_path + obj_name, 'wb') as f:
        pickle.dump(obj, f)

# Read serialized objects in serialized/
def read(obj_name):
    if not os.path.exists(serialized_dir):
        os.mkdir(serialized_dir)
    if obj_name == "SNPdata.ser":
        return read_feather(serialized_dir + obj_name)
    
    dir_path = serialized_dir
    if not os.path.isfile(dir_path + obj_name):
        print("No such file exists.")
        return None
    with open(dir_path + obj_name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_ticker_list():
    return read("SNPdata.ser")["Ticker"].unique().tolist()

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def get_text_categories():
    out = ["Ticker"]
    return out

def get_floats_categories():
    raise NotImplementedError("get_window_categories() is not implemented yet.")
    categories = ["TypicalPrice", "Volume"]
    days_back = [10, 20, 40, 80, 160, 320, 640, 1280]
    output = []

    for cat in categories:
        for i in range(len(days_back)):
            if i == 0:
                output.append(f"{cat}_0_{days_back[i]}_days")
            else:
                output.append(f"{cat}_{days_back[i-1]}_{days_back[i]}_days")

    return output

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_num_days(window_category_name):
    parts = window_category_name.split('_')
    start = int(parts[-3])
    end = int(parts[-2])
    num_days = end - start
    return num_days

def get_time2vec_categories():
    categories = ['Year', 'Month', 'Day', 'Weekday', 'Hour', 'Minute']
    return categories

def get_text_categories():
    out = ["Ticker"]
    return out

def base_categories():
    categories = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return categories

def get_floats_categories():
    categories = base_categories()
    days_back = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2584, 3072]
    output = []

    for cat in categories:
        for i in range(len(days_back)):
            if i == 0:
                output.append(f"{cat}_0_{days_back[i]}_days")
            else:
                output.append(f"{cat}_{days_back[i-1]}_{days_back[i]}_days")

    return output

def get_base_category_name(window_category_name):
    parts = window_category_name.split('_')
    category = parts[0]
    return category

def percent_error(gt_val, pred_val):
    assert pred_val.shape == gt_val.shape, f"pred_val.shape: {pred_val.shape}, gt_val.shape: {gt_val.shape}"
    assert torch.all(gt_val.gt(-1e-8)), f"gt_val: {gt_val}"
    return ((gt_val - pred_val) / (gt_val + 1e-8))

def mean_squared_error(gt_val, pred_val):
    assert pred_val.shape == gt_val.shape, f"pred_val.shape: {pred_val.shape}, gt_val.shape: {gt_val.shape}"
    return (gt_val - pred_val) ** 2

