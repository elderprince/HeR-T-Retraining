import sys
from pathlib import Path

# in jupyter (lab / notebook), based on notebook path
module_path = str(Path.cwd().parents[0])

# in standard python
# module_path = str(Path.cwd(__file__).parents[0] / "py")

if module_path not in sys.path:
    sys.path.append(module_path)

import wandb

import data.donut_dataset as donut_dataset
import utils.helpers as helpers
import utils.confidential as confidential
import utils.push_to_hub as push_to_hub
import pytorch_lightning as pl

from datasets import load_dataset
from torch.utils.data import DataLoader
from data.donut_dataset import DonutDataset
from models.donut_pytorch_lightning import DonutModelPLModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

image_path = "/Users/WilliamLiu/HeR_T_retaining/data/img_test"
max_length = 768

dataset = donut_dataset.data_loader(image_path)
print('Data Loading completes.')

processor, model = donut_dataset.model_loader(dataset, max_length, 
                                              "naver-clova-ix/donut-base")
print('Processor and Model are loaded.')

processor.image_processor.size = helpers.image_size(dataset)
processor.image_processor.do_align_long_axis = False
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_herbarium>'])[0]
model.config.hidden_dropout_prob = 0.2
model.config.attention_probs_dropout_prob = 0.2
print("Model and processor settings are completed.")

train_dataset = DonutDataset(image_path, max_length=max_length,
                             split="train", task_start_token="<s_herbarium>", 
                             prompt_end_token="<s_herbarium>",
                             sort_json_key=False, # cord dataset is preprocessed -> no need for this
                             model=model, 
                             processor=processor,
                             )
val_dataset = DonutDataset(image_path, max_length=max_length,
                             split="validation", task_start_token="<s_herbarium>", 
                             prompt_end_token="<s_herbarium>",
                             sort_json_key=False, # cord dataset is preprocessed -> no need for this
                             model=model,
                             processor=processor,
                             )
print("DonutDataset is loaded.")

config = {
    'max_epochs': 20,
    'val_check_interval': 0.25,
    'check_val_every_n_epoch': 1,
    'gradient_clip_val': 1.0,
    'num_training_samples_per_epoch': 36760,
    'lr': 2.5e-5, # or 2e-5
    'weight_decay': 2e-5,
    'dropout_rate': 0.2,
    'train_batch_sizes': 8,
    'val_batch_sizes': 1,
    'num_nodes': 1,
    'warmup_steps': 2500,
    'result_path': "/Users/WilliamLiu/HeR_T_retaining/results",
    'verbose': True, 
    'seed': 16, 
    'num_workers': 2
}

model_lightning = DonutModelPLModule(config, processor, model, train_dataset, val_dataset)
print('PyTorch Lightning Model has been set up.')

# api key so that it doesn't ask me for it
wandb.login(key=confidential.api_key)
wandb_logger = WandbLogger(project="HeR-T-trial", name="localTrial")
# use default patiente
early_stop_callback = EarlyStopping(monitor="val_edit_distance", verbose=True, mode="min")

pushToHub = push_to_hub.PushToHubCallback()
trainer = pl.Trainer(
        accelerator="mps",
        #accelerator="gpu",
        devices=1,
        # strategy="xla_debug",
        max_epochs=config['max_epochs'],
        val_check_interval=config['val_check_interval'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        gradient_clip_val=config['gradient_clip_val'],
        precision='bf16', # we'll use mixed precision
        #precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[pushToHub, early_stop_callback]
)
trainer.fit(model_lightning)