import fasttext


model_path = "/veld/data/veld_data_8_fasttext_models/data/test/test.bin"
model = fasttext.load_model(model_path)

word = "Europa"
word_vector = model.get_word_vector(word)

print(word_vector)
