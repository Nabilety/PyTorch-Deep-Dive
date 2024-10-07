# ## Higher-level PyTorch APIs: a short introduction to PyTorch Lightning

# ### Setting up the PyTorch Lightning model

# ## Higher-level PyTorch APIs: a short introduction to PyTorch Lightning

# ### Setting up the PyTorch Lightning model


# Lighting makes training deep neural network simpler by removing much of the boilerplate
# code. However, while the focus lies in simplicity and flexibility, it also allows us to use many
# advanced features such as multi-GPU support and fast low-precision training

# As the classifying MNIST project, where we implemented a multilayer perceptron for classifying
# handwritten digits in the MNIST dataset, we will reimplement this classifier using Lightning.

# Use LightningModule instead of regular PyTorch module to implement a Lightning model
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics import __version__ as torchmetrics_version
from pkg_resources import parse_version

class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()

        # new PL attributes:
        # Allows us to track accuracy during training
        if parse_version(torchmetrics_version) > parse_version("0.8"):
            self.train_acc = Accuracy(task="multiclass", num_classes=10)
            self.valid_acc = Accuracy(task="multiclass", num_classes=10)
            self.test_acc = Accuracy(task="multiclass", num_classes=10)
        else:
            self.train_acc = Accuracy()
            self.valid_acc = Accuracy()
            self.test_acc = Accuracy()

        # Model similar to previous section:
        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit

        all_layers.append(nn.Linear(hidden_units[-1], 10))
        self.model = nn.Sequential(*all_layers)

    # Simple forward pass that returns the logits, when we call our model on the input data
    # (logits: outputs of the last fully connected layer of our network before softmax layer)
    # logits computed via the forward method by calling self(x), used for train, validation and test steps
    def forward(self, x):
        x = self.model(x)
        return x

    # train/validation/test_step, train/validation/test_epoch and configure_optimizers methods are
    # built-in methods recognized by Lightning.
    # i.e. training_step defines a single forward pass during training, where we also keep track of
    # the accuracy and loss that we can analyze later.
    # training_step method is executed on each individual batch during training, and via training_epoch_end
    # exectued at the end of each training epoch, we compute the training set accuracy from the accuracy values accumulated via training
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # simply put similar to pred = model(x_batch)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y) # note we compute accuracy here, but don't log it yet
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    # specify optimizer used for training
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# ### Setting up the data loaders
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
# There are three main ways in which we can prepare the dataset for Lightning. We can:
# • Make the dataset part of the model
# • Set up the data loaders as usual and feed them to the fit method of a Lightning Trainer—the
# Trainer is introduced in the next subsection
# • Create a LightningDataModule

# We will use LightningDataModule, which is more organized approach
# This consist of five main methods:
class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    # Define general steps, such as downloading dataset
    def prepare_data(self):
        MNIST(root=self.data_path, download=True)

    # Define datasets used for training, validation and testing
    # Note MNIST does not have dedicated validation split, which is why we use random_split function
    # to divide the 60.000 example training set into 55.000 examples for training and 5.000 for validation
    def setup(self, stage=None):
        # stage is either 'fit', 'validate', 'test', or 'predict'
        # here note relevant
        mnist_all = MNIST(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )

        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )

    # data loader define how the respective datasets are loaded
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, persistent_workers=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=4)

def main():
    # Initialize data module and use it for training, validation and testing
    torch.manual_seed(1)
    mnist_dm = MnistDataModule()

    # ### Training the model using the PyTorch Lightning Trainer class

    # Trainer class makes the training model super convenient by taking care of all the intermediate steps
    # such as calling zero_grad(), backward(), and optimizer.step() for us. Bonus, it lets us specify one
    # or more GPUs to use (if available):
    mnistclassifier = MultiLayerPerceptron()

    callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="valid_acc")]  # save top 1 model

    if torch.cuda.is_available():  # if you have GPUs
        trainer = pl.Trainer(max_epochs=10, callbacks=callbacks, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=10, callbacks=callbacks)

    trainer.fit(model=mnistclassifier, datamodule=mnist_dm)

    # preceding code train our multilayer perceptron for 10 epochs.
    # During training we see a handy progress bar keeping track of the epoch and core metrics
    # such as training and validation losses. After training finished we can inspect metrics we logged:

    # ### Evaluating the model using TensorBoard

    trainer.test(model=mnistclassifier, datamodule=mnist_dm, ckpt_path='best')

    # Start tensorboard

    path = 'lightning_logs/version_0/checkpoints/epoch=8-step=7739.ckpt'

    # PyTorch Lightning (v1.5.0+), the argument resume_from_checkpoint has been deprecated from the Trainer class
    # so we replace resume_from_checkpoint with ckpt_path argument serving the same purpose, and pass it directly into
    # the fit() method instead of the Trainer below.
    if torch.cuda.is_available():  # if you have GPUs
        trainer = pl.Trainer(
            max_epochs=15, callbacks=callbacks, gpus=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=15, callbacks=callbacks
        )

    trainer.fit(model=mnistclassifier, datamodule=mnist_dm, ckpt_path=path)

    trainer.test(model=mnistclassifier, datamodule=mnist_dm)

    trainer.test(model=mnistclassifier, datamodule=mnist_dm, ckpt_path='best')

    path = "lightning_logs/version_0/checkpoints/epoch=13-step=12039.ckpt"
    model = MultiLayerPerceptron.load_from_checkpoint(path)

if __name__ == '__main__':
    main()





