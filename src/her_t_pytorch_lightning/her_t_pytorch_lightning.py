import re
import torch
import random
import numpy as np
import pytorch_lightning as pl

from nltk import edit_distance
from torch.utils.data import DataLoader

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor
        dropout_rate = config['dropout_rate']  # Access the dropout_rate from the config
        model.config.hidden_dropout_prob = dropout_rate
        model.config.attention_probs_dropout_prob = dropout_rate
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.g = torch.Generator()
        self.g.manual_seed(config['seed'])
        # save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch
        
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), 
                                       self.model.config.decoder_start_token_id, 
                                       device=self.device)
        
        outputs = self.model.generate(pixel_values,
                                      decoder_input_ids=decoder_input_ids,
                                      max_length=768,
                                      early_stopping=False,
                                      pad_token_id=self.processor.tokenizer.pad_token_id,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      num_beams=1,
                                      bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True,)
    
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, 
                              "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        accuracies = []  # Track validation accuracies
        
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))
            
            if self.config.get("verbose", False) and len(scores) == 1:
                pass
                # print(f"Prediction: {pred}")
                # print(f"    Answer: {answer}")
                # print(f" Normed ED: {scores[0]}")

            # Calculate accuracy and append to accuracies list
            accuracy = 1 - scores[-1]  # Subtract edit distance from 1 to get accuracy
            accuracies.append(accuracy)

        self.log("val_edit_distance", np.mean(scores))
        self.log("val_accuracy", np.mean(accuracies))  # Log the mean validation accuracy

        return scores
    
    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
    
        return optimizer
    
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.config['train_batch_sizes'],
            # num_workers=self.config['num_workers'],
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
            shuffle=True,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            self.config['val_batch_sizes'],
            pin_memory=True,
            shuffle=False,
        )
        
        return val_loader

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)