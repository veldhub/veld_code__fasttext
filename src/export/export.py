import pickle
import os

import fasttext


IN_MODEL_FILE = "/veld/input/" + os.getenv("in_model_file")
OUT_VECTOR_FILE = "/veld/output/" + os.getenv("out_vector_file")
print("IN_MODEL_FILE:", IN_MODEL_FILE)
print("OUT_VECTOR_FILE:", OUT_VECTOR_FILE)


# loading model
print("loading model")
model = fasttext.load_model(IN_MODEL_FILE)

# transforming vectors to dict
print("transforming vectors to dict")
vector_dict = {}
for word in model.get_words():
    vector_dict[word] = model.get_word_vector(word)

# persisting dict to pickle
print("persisting dict to pickle")
with open(OUT_VECTOR_FILE, "wb") as f:
    pickle.dump(vector_dict, f)

