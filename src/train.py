import fasttext
import os
import subprocess
import yaml
from datetime import datetime


# train data 
TRAIN_DATA_PATH = os.getenv("train_data_path")
TRAIN_DATA_DESCRIPTION = os.getenv("train_data_description")

# model data
TRAINING_ARCHITECTURE = os.getenv("training_architecture")
MODEL_PATH = os.getenv("model_path")
MODEL_ID = MODEL_PATH.split("/")[-1]
MODEL_METADATA_PATH = os.getenv("out_metadata_path")

# model hyperparameters
VECTOR_SIZE = int(os.getenv("vector_size"))
EPOCHS = int(os.getenv("epochs"))

# training process metadata
DURATION = None


def print_params():
    print(f"TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
    print(f"TRAIN_DATA_DESCRIPTION: {TRAIN_DATA_DESCRIPTION}")
    print(f"TRAINING_ARCHITECTURE: {TRAINING_ARCHITECTURE}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_ID: {MODEL_ID}")
    print(f"MODEL_METADATA_PATH: {MODEL_METADATA_PATH}")
    print(f"VECTOR_SIZE: {VECTOR_SIZE}")
    print(f"EPOCHS: {EPOCHS}")


def train_and_persist():
    time_start = datetime.now()
    # flush necessary for jupyter VM podman, to keep prints synchronized
    print("training start:", time_start, flush=True)
    model = fasttext.train_unsupervised(TRAIN_DATA_PATH, epoch=EPOCHS, dim=VECTOR_SIZE)
    time_end = datetime.now()
    print("training done:", time_end, flush=True)
    global DURATION
    DURATION = (time_end - time_start).seconds / 3600
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model.save_model(MODEL_PATH + "/model.bin")


def write_metadata():

    # calculate size of training and model data
    def calc_size(file_or_folder):
        size = subprocess.run(["du", "-sh", file_or_folder], capture_output=True, text=True)
        size = size.stdout.split()[0]
        return size
    train_data_size = calc_size(TRAIN_DATA_PATH)
    model_data_size = calc_size(MODEL_PATH)

    # calculate hash of training data
    train_data_md5_hash = subprocess.run(["md5sum", TRAIN_DATA_PATH], capture_output=True, text=True)
    train_data_md5_hash = train_data_md5_hash.stdout.split()[0]

    # aggregate into metadata dictionary
    metadata = {
        "training_architecture": TRAINING_ARCHITECTURE,
        "model_id": MODEL_ID, 
        "train_data_name": TRAIN_DATA_DESCRIPTION,
        "train_data_size": train_data_size,
        "train_data_md5_hash": train_data_md5_hash,
        "training_epochs": EPOCHS,
        "training_vector_size": VECTOR_SIZE,
        "training_duration (hours)": round(DURATION, 1),
        "model_data_size": model_data_size,
    }

    # write to yaml
    with open(MODEL_METADATA_PATH, "w") as f:
        # iteration over dictionary to ensure the yaml writer respects the order
        for k, v in metadata.items():
            yaml.dump({k: v}, f)


def main():
    print_params()
    train_and_persist()
    write_metadata#()


if __name__ == "__main__":
    main()

