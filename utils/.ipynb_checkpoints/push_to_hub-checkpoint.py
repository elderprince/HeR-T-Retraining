import torch
import os
import pytorch_lightning as pl

class PushToHubCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        save_dir = "/leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Retraining/checkpoints/600x800"
        print(f"Saving model to the local, epoch {trainer.current_epoch}")
        pl_module.processor.save_pretrained(os.path.join(save_dir, 'processor-epoch-{}'.format(trainer.current_epoch)), 
                                            push_to_hub = False)
        pl_module.model.save_pretrained(os.path.join(save_dir, 'model-epoch-{}'.format(trainer.current_epoch)), 
                                            push_to_hub = False)
        torch.save(pl_module.model.state_dict(), os.path.join(save_dir, 'model-states-epoch-{}.pth'.format(trainer.current_epoch)))
    def on_train_end(self, trainer, pl_module):
        result_dir = "/leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Retraining/results/600x800"
        print(f"Saving model to the local after training")
        pl_module.processor.save_pretrained(os.path.join(result_dir, 'processor'), push_to_hub = False)
        pl_module.model.save_pretrained(os.path.join(result_dir, 'model'), push_to_hub = False)
        torch.save(pl_module.model.state_dict(), os.path.join(result_dir, 'model-states.pth'))