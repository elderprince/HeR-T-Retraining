from datasets import load_dataset

def data_loader(img_path): 
    dataset = load_dataset(img_path)
    print(f'this is the dataset {dataset}')

    return dataset