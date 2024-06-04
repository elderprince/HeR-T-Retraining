import random

def random_example(dataset): 
    # Create a random seed to select a sample
    random_num = random.randint(0, len(dataset['train']))
    
    # Extract the sample and its ground truth
    sample = dataset['train'][random_num]
    image = sample['image']
    ground_truth = sample['ground_truth']

    return image, ground_truth

def image_size(dataset):
    img, gt = random_example(dataset)
    width, height = img.size
    
    return width, height