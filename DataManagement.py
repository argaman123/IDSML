import pickle

# loads any object from a file
def load(filename: str):
    file = open(filename, "rb")
    return pickle.load(file)

# stores any object in a file
def save(filename: str, object):
    with open(filename, "wb") as f:
        pickle.dump(object, f)

