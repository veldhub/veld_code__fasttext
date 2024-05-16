#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fasttext
import os
import subprocess
import yaml
from datetime import datetime


# ## config

# In[ ]:


# train data 
TRAIN_DATA_PATH = os.getenv("train_data_path")
TRAIN_DATA_DESCRIPTION = os.getenv("train_data_description")

# model data
TRAINING_ARCHITECTURE = os.getenv("training_architecture")
MODEL_ID = os.getenv("model_id")
MODEL_PATH = os.getenv("model_path") + MODEL_ID + "/"
MODEL_METADATA_PATH = MODEL_PATH + "metadata.yaml"

# model hyperparameters
VECTOR_SIZE = int(os.getenv("vector_size"))
EPOCHS=int(os.getenv("epochs"))


# In[ ]:


print(f"TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"TRAIN_DATA_DESCRIPTION: {TRAIN_DATA_DESCRIPTION}")
print(f"TRAINING_ARCHITECTURE: {TRAINING_ARCHITECTURE}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"MODEL_ID: {MODEL_ID}")
print(f"MODEL_METADATA_PATH: {MODEL_METADATA_PATH}")
print(f"VECTOR_SIZE: {VECTOR_SIZE}")
print(f"EPOCHS: {EPOCHS}")


# ## training and persisting

# In[ ]:


time_start = datetime.now()
model = fasttext.train_unsupervised(TRAIN_DATA_PATH, epoch=EPOCHS, dim=VECTOR_SIZE)
time_end = datetime.now()
duration = (time_end - time_start).seconds / 60
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
model.save_model(MODEL_PATH + "/model.bin")


# ## writing metadata 

# In[ ]:


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
    "training_vector_size": VECTOR_SIZE,
    "epochs": EPOCHS,
    "training_duration (minutes)": round(duration, 1),
    "model_data_size": model_data_size,
}

# write to yaml
with open(MODEL_METADATA_PATH, "w") as f:
    # iteration over dictionary to ensure the yaml writer respects the order
    for k, v in metadata.items():
        yaml.dump({k: v}, f)

