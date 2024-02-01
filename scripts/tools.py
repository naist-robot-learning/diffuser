import pickle
import numpy as np

def dict2csv(data: dict):
    
    # Specify the file path
    pickle_file_path = 'my_dict.pickle'
    
    # Writing to a pickle file
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
        
    print(f'Dictionary has been stored to {pickle_file_path}.')
    
