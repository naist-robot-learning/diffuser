import pickle
import numpy as np

def dict2pickle(data: dict):
    
    # Specify the file path
    pickle_file_path = 'my_dict.pickle'
    
    # Writing to a pickle file
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
        
    print(f'Dictionary has been stored to {pickle_file_path}.')
    
def pickle2dict(filepath: str) -> dict:
    
    # Reading from the pickle file
    with open(filepath, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    
    print('Loaded Dictionary:')
    return loaded_dict
    