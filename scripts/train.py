import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from her_t_pytorch_lightning import (
    dataset_loader,
    utils,
    DonutModelPLModule
    )

from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel
    )

kwargs = utils.read_config('experiments/config')

image_path = kwargs['image_path']
max_length = kwargs['max_length']
pretrain_model = kwargs['pretrain_model']
project_name = kwargs['project_name']
log_name = kwargs['log_name']

dataset = dataset_loader.data_loader(image_path)
print('Data Loading completes.')

processor = DonutProcessor.from_pretrained(pretrain_model)
model = VisionEncoderDecoderModel.from_pretrained(pretrain_model)
print('Processor and Model are loaded.')

processor.image_processor.size = utils.image_size(dataset)
processor.image_processor.do_align_long_axis = False
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_herbarium>'])[0]
model.config.hidden_dropout_prob = 0.2
model.config.attention_probs_dropout_prob = 0.2
print("Model and processor settings are completed.")

train_dataset = dataset_loader.DonutDataset(image_path, 
                                            max_length=max_length,
                                            split="train", 
                                            task_start_token="<s_herbarium>", 
                                            prompt_end_token="<s_herbarium>",
                                            # cord dataset is preprocessed -> no need for this
                                            sort_json_key=False, 
                                            model=model, 
                                            processor=processor,
                                            )
val_dataset = dataset_loader.DonutDataset(image_path, 
                                          max_length=max_length,
                                          split="validation", 
                                          task_start_token="<s_herbarium>", 
                                          prompt_end_token="<s_herbarium>",
                                          # cord dataset is preprocessed -> no need for this
                                          sort_json_key=False, 
                                          model=model,
                                          processor=processor,
                                          )
print("DonutDataset is loaded.")

pl_config = utils.read_config('experiments/pl_config')
model_lightning = DonutModelPLModule(pl_config, processor, model, 
                                     train_dataset, val_dataset)
print('PyTorch Lightning Model has been set up.')

wandb.init(mode="offline")
wandb_logger = WandbLogger(project=project_name, name=log_name)
early_stop_callback = EarlyStopping(monitor="val_edit_distance", 
                                    verbose=True, mode="min", patience=7)
pushToHub = utils.PushToHubCallback()

trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,
        # strategy="xla_debug",
        max_epochs=config['max_epochs'],
        val_check_interval=config['val_check_interval'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        gradient_clip_val=config['gradient_clip_val'],
        precision='bf16-mixed', # we'll use mixed precision
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[pushToHub, early_stop_callback]
)

print("Training starts.")
trainer.fit(model_lightning)