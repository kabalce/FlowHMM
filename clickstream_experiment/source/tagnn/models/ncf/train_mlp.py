import ast

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodules import MovieLensDataModule
from src.models.ncf.mlp import MultiLayerPerceptron


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.option('--name', type=click.STRING)
@click.option('--dataset', type=click.STRING, default='movielens')
@click.option('--variant', type=click.STRING, default='1m')
@click.option('--dataset_path', type=click.STRING, default=None)
@click.option('--layer_sizes', cls=PythonLiteralOption, default=[128, 64, 32])
@click.option('--epochs', type=click.INT, default=256)
@click.option('--batch_size', type=click.INT, default=4096)
@click.option('--validation/--no_validation', default=True)
@click.option('--test/--no_test', default=False)
@click.option('--num_negatives', type=click.INT, default=8)
@click.option('--lr', type=click.FLOAT, default=1e-3)
@click.option('--lr_decay', type=click.FLOAT, default=0.1)
@click.option('--lr_step_size', type=click.INT, default=16)
@click.option('--weight_decay', type=click.FLOAT, default=0)
@click.option('--embeddings/--no_embeddings', default=False)
@click.option('--unfreeze_after', type=click.INT, default=8)
@click.option('--save_path', type=click.STRING)
def main(
        name, dataset, variant, dataset_path, layer_sizes, epochs, batch_size, validation, test, num_negatives, lr, lr_decay,
        lr_step_size, weight_decay, embeddings, unfreeze_after, save_path
):
    wandb_logger = WandbLogger(project="MultiLayerPerceptron", entity="cirglaboratory", name=name)

    if dataset == 'movielens':
        datamodule = MovieLensDataModule(
            variant=variant, batch_size=batch_size, validate=validation, n_negatives=num_negatives,
            dataset_path=dataset_path,
        )
    else:
        raise NotImplementedError

    model = MultiLayerPerceptron(
        datamodule.n_users,
        datamodule.n_items,
        layer_sizes,
        lr=lr,
        lr_decay=lr_decay,
        lr_step_size=lr_step_size,
        weight_decay=weight_decay,
        init_embeddings=embeddings,
        unfreeze_after=unfreeze_after,
    )
    model.datamodule = datamodule

    print("Starting training")

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        accelerator='auto',
        devices=1,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
    )
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    if test:
        trainer.test(datamodule=datamodule)
    if save_path is not None:
        torch.save(model.mlp.state_dict(), save_path + '/mlp.pt')


if __name__ == '__main__':
    main()
