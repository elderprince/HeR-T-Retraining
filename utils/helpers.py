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
    img_size = img.size
    
    return img_size

class PushToHubCallback():
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.processor.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training in progress, 
                                    epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training in progress, 
                                    epoch {trainer.current_epoch}")
    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training done")