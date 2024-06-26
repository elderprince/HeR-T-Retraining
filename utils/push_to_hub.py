import pytorch_lightning as pl

class PushToHubCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.processor.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")
    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub("Jac-Zac/thesis_donut",
                                    commit_message=f"Training done")