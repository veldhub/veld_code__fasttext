import fasttext
import os
import subprocess
import yaml
from datetime import datetime


# train data 
TRAIN_DATA_FILE = os.getenv("in_train_data_file")
TRAIN_DATA_PATH = "/veld/input/" + TRAIN_DATA_FILE

# model data
TRAINING_ARCHITECTURE = "fasttext"
MODEL_DESCRIPTION = os.getenv("model_description")
OUT_MODEL_FILE = os.getenv("out_model_file")
OUT_MODEL_PATH = "/veld/output/" + OUT_MODEL_FILE
MODEL_ID = OUT_MODEL_FILE.replace(".bin", "")
OUT_MODEL_METADATA_PATH = "/veld/output/veld.yaml"

# model hyperparameters
VECTOR_SIZE = int(os.getenv("vector_size"))
EPOCHS = int(os.getenv("epochs"))
WINDOW_SIZE = int(os.getenv("window_size"))

# dynamically loaded metadata
TRAIN_DATA_DESCRIPTION = None
DURATION = None


def get_desc():
    veld_file = None
    for file in os.listdir("/veld/input/"):
        if file.startswith("veld") and file.endswith("yaml"):
            if veld_file is not None:
                raise Exception("Multiple veld yaml files found.")
            else:
                veld_file = file
    if veld_file is None:
        print("no training data veld yaml file found. Won't be able to persist that as metadata.", flush=True)
    else:
        with open("/veld/input/" + veld_file, "r") as f:
            input_veld_metadata = yaml.safe_load(f)
            global TRAIN_DATA_DESCRIPTION
            try:
                TRAIN_DATA_DESCRIPTION = input_veld_metadata["x-veld"]["data"]["description"]
            except:
                pass


def train_and_persist():
    time_start = datetime.now()
    print("training start:", time_start, flush=True)
    model = fasttext.train_unsupervised(
        TRAIN_DATA_PATH, 
        epoch=EPOCHS, 
        dim=VECTOR_SIZE, 
        verbose=3,
        ws=WINDOW_SIZE,
    )
    time_end = datetime.now()
    print("training done:", time_end, flush=True)
    global DURATION
    DURATION = (time_end - time_start).seconds / 60
    model.save_model(OUT_MODEL_PATH)


def write_metadata():

    # calculate size of training and model data
    def calc_size(file_or_folder):
        size = subprocess.run(["du", "-sh", file_or_folder], capture_output=True, text=True)
        size = size.stdout.split()[0]
        return size
    train_data_size = calc_size(TRAIN_DATA_PATH)
    model_data_size = calc_size(OUT_MODEL_PATH)

    # calculate hash of training data
    train_data_md5_hash = subprocess.run(["md5sum", TRAIN_DATA_PATH], capture_output=True, text=True)
    train_data_md5_hash = train_data_md5_hash.stdout.split()[0]

    # aggregate into metadata dictionary
    out_veld_metadata = {
        "x-veld": {
            "data": {
                "description": MODEL_DESCRIPTION,
                "file_type": "bin",
                "contents": [
                    "word embeddings model",
                    "fasttext model",
                ],
                "additional": {
                    "model_id": MODEL_ID,
                    "train_data_description": TRAIN_DATA_DESCRIPTION,
                    "training_architecture": TRAINING_ARCHITECTURE,
                    "train_data_size": train_data_size,
                    "train_data_md5_hash": train_data_md5_hash,
                    "training_epochs": EPOCHS,
                    "training_vector_size": VECTOR_SIZE,
                    "training_window_size": WINDOW_SIZE,
                    "training_duration (minutes)": round(DURATION, 1),
                    "model_data_size": model_data_size,
                }
            }
        }
    }

    # write to yaml
    with open(OUT_MODEL_METADATA_PATH, "w") as f:
        yaml.dump(out_veld_metadata, f, sort_keys=False)


def main():
    get_desc()
    train_and_persist()
    write_metadata()


if __name__ == "__main__":
    main()

