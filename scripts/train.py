import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from her_t_pytorch_lightning.her_t_pytorch_lightning import DonutModelPLModule

from her_t_pytorch_lightning import (
    utils,
    dataset_loader
    )

from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel
    )

kwargs = utils.read_config('experiments/train_config')

image_path = kwargs['image_path']
max_length = kwargs['max_length']
pretrained_model = kwargs['pretrained_model']
pretrained_processor = kwargs['pretrained_processor']
save_dir = kwargs['save_dir']
result_dir = kwargs['result_dir']
result_path = kwargs['result_path']

dataset = dataset_loader.data_loader(image_path)
print('Data Loading completes.')

processor = DonutProcessor.from_pretrained(pretrained_processor)
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model)
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

pl_config = {
    'max_epochs': 20, 
    'val_check_interval': 0.25, 
    'check_val_every_n_epoch': 1, 
    'gradient_clip_val': 1.0, 
    'num_training_samples_per_epoch': 36760, 
    'lr': 2.5e-5,  # or 2e-5
    'weight_decay': 2e-5, 
    'dropout_rate': 0.2, 
    'train_batch_sizes': 8, 
    'val_batch_sizes': 8, 
    'num_nodes': 1, 
    'warmup_steps': 2500, 
    'result_path': result_path,
    'verbose': True, 
    'seed': 16, 
    'num_workers': 1
}
model_lightning = DonutModelPLModule(pl_config, processor, model, 
                                     train_dataset, val_dataset)
print('PyTorch Lightning Model has been set up.')

early_stop_callback = EarlyStopping(monitor="val_edit_distance", 
                                    verbose=True, mode="min", patience=7)
pushToHub = utils.PushToHubCallback(save_dir, result_dir)

trainer = pl.Trainer(
    accelerator='cuda',
    strategy='ddp',
    devices=4,
    # strategy="xla_debug",
    max_epochs=pl_config['max_epochs'],
    val_check_interval=pl_config['val_check_interval'],
    check_val_every_n_epoch=pl_config['check_val_every_n_epoch'],
    gradient_clip_val=pl_config['gradient_clip_val'],
    precision='bf16-mixed', # we'll use mixed precision
    num_sanity_val_steps=0,
    callbacks=[pushToHub, early_stop_callback]
    )
    

print("Training starts.")
trainer.fit(model_lightning)