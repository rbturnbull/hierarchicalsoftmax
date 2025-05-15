# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CIFAR Demonstration

    This notebook demonstrates how to use the `hierarchicalsoftmax` module to train a neural network on the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First, choose the hyperparameters.""")
    return


@app.cell
def _(mo):
    cifar_radio = mo.ui.radio(options=["10","100"], value=mo.cli_args().get("cifar") or "100", label="CIFAR Dataset")
    batch_size_input = mo.ui.number(value=mo.cli_args().get("batch") or 32, label="Batch Size")
    epochs_input = mo.ui.number(value=mo.cli_args().get("batch") or 30, label="Epochs")
    mo.vstack([cifar_radio, epochs_input, batch_size_input])
    return batch_size_input, cifar_radio, epochs_input


@app.cell
def _(batch_size_input, cifar_radio, epochs_input):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    assert cifar_radio.value in ["10","100"]
    batch_size = batch_size_input.value
    epochs = epochs_input.value
    cifar_dataset = datasets.CIFAR10 if cifar_radio.value == "10" else datasets.CIFAR100

    # Use the same data augmentation strategies as in https://arxiv.org/pdf/1605.07146v4
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_data = cifar_dataset(root=".", train=True, download=True, transform=transform)
    test_data = cifar_dataset(root=".", train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return (
        DataLoader,
        batch_size,
        epochs,
        test_data,
        test_loader,
        train_data,
        train_loader,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot the first 10 images""")
    return


@app.cell
def _(train_data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    num_images = 10

    # Create a row of subplots
    cifar_fig = make_subplots(
        rows=1, cols=num_images, 
        subplot_titles=[train_data.classes[train_data[i][1]] for i in range(num_images)], 
        horizontal_spacing=0,
    )

    for i in range(num_images):
        img, label = train_data[i]
        img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C) and convert to numpy

        cifar_fig.add_trace(
            go.Image(z=(img * 255).astype('uint8')),
            row=1, col=i+1
        )

    # Update layout: remove axes and tighten spacing
    thumbnail_size = 105
    cifar_fig.update_layout(
        height=thumbnail_size,  # adjust height as needed
        width=thumbnail_size * num_images,  # 150px per image
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Hide axes
    for i in range(1, num_images + 1):
        cifar_fig.update_xaxes(visible=False, row=1, col=i)
        cifar_fig.update_yaxes(visible=False, row=1, col=i)

    cifar_fig
    return (go,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Non-hierarchical model

    First we create a basic non-hierarchical model as a baseline
    """
    )
    return


@app.cell
def _(train_data):
    import torch
    from torch import nn
    import torch.nn.functional as F
    import lightning as L
    from torchmetrics import Accuracy

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride, dropout_rate=0.3):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != out_planes:
                self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        def forward(self, x):
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.dropout(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += self.shortcut(x)
            return out


    class WideResNetBody(nn.Module):
        def __init__(self, depth:int=28, width_factor:int=10, dropout_rate:float=0.3):
            super().__init__()
            assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
            n = (depth - 4) // 6

            k = width_factor
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

            self.layer1 = self._make_layer(16*k, n, stride=1, dropout_rate=dropout_rate)
            self.layer2 = self._make_layer(32*k, n, stride=2, dropout_rate=dropout_rate)
            self.layer3 = self._make_layer(64*k, n, stride=2, dropout_rate=dropout_rate)

            self.bn = nn.BatchNorm2d(64*k)

        def _make_layer(self, out_planes, blocks, stride, dropout_rate):
            strides = [stride] + [1]*(blocks-1)
            layers = []
            for s in strides:
                layers.append(BasicBlock(self.in_planes, out_planes, s, dropout_rate))
                self.in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            return out


    class BasicImageClassifier(L.LightningModule):
        def __init__(self, depth:int=28, width_factor:int=10, dropout_rate:float=0.3):
            super().__init__()
            self.model = nn.Sequential(
                WideResNetBody(depth=depth, width_factor=width_factor, dropout_rate=dropout_rate),
                nn.LazyLinear(out_features=len(train_data.classes))
            )
            self.loss_fn = nn.CrossEntropyLoss()
            self.metrics = [
                Accuracy(task="multiclass", num_classes=len(train_data.classes))
            ]

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.loss_fn(logits, y)
            self.log('train_loss', loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.loss_fn(logits, y)
            self.log('val_loss', loss, prog_bar=True)
            for metric in self.metrics:
                metric = metric.to(logits.device)
                result = metric(logits, y)
                if isinstance(result, dict):
                    for name, value in result.items():
                        self.log(f"val_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
                else:
                    self.log(f"val_{metric.__class__.__name__}", result, on_step=False, on_epoch=True, prog_bar=True)

            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    basic_model = BasicImageClassifier(depth=16, width_factor=8, dropout_rate=0.3)
    basic_model
    return BasicImageClassifier, L, WideResNetBody, basic_model, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train the basic model""")
    return


@app.cell
def _(L, basic_model, epochs, test_loader, train_loader):
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import TQDMProgressBar

    basic_logger = CSVLogger(save_dir="lightning_logs", name="basic_model")
    basic_trainer = L.Trainer(max_epochs=epochs, accelerator="auto", enable_checkpointing=False, logger=basic_logger, callbacks=[TQDMProgressBar(leave=True, refresh_rate=20)])
    basic_trainer.fit(basic_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    return CSVLogger, TQDMProgressBar, basic_logger


@app.cell
def _(basic_logger, go):
    import pandas as pd
    from pathlib import Path
    import plotly.io as pio

    pio.templates.default = "plotly_white"

    basic_metrics_df = pd.read_csv(Path(basic_logger.log_dir) / "metrics.csv")
    basic_metrics_df = basic_metrics_df.dropna(subset=["val_MulticlassAccuracy"])
    basic_fig = go.Figure()
    basic_fig.add_trace(go.Scatter(x=basic_metrics_df["epoch"], y=basic_metrics_df["val_MulticlassAccuracy"], mode='lines', name='class'))
    basic_fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
    )
    basic_fig.write_html(str(Path(basic_logger.log_dir)/"accuracy.html"))
    basic_fig.show()

    return Path, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Hierarchical model

    Let's now create a hierarchical model.
    First we need to create a tree structure for the CIFAR dataset.
    """
    )
    return


@app.cell
def _(mo, train_data):
    from hierarchicalsoftmax import (
        SoftmaxNode,
        HierarchicalSoftmaxLazyLinear,
        HierarchicalSoftmaxLoss,
    )

    if len(train_data.classes) == 10:
        # CIFAR-10
        superclasses = {
            "animals": ["bird", "cat", "deer", "dog", "frog", "horse"],
            "vehicles": ["airplane", "automobile", "ship", "truck"],
        }
    else:
        # CIFAR-100
        superclasses = {
            "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food containers": ["bottle", "bowl", "can", "cup", "plate"],
            "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
            "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
            "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
            "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
            "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
            "people": ["baby", "boy", "girl", "man", "woman"],
            "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        }


    root = SoftmaxNode("root")
    for superclass, classes in superclasses.items():
        superclass_node = SoftmaxNode(superclass, parent=root)
        for class_name in classes:
            SoftmaxNode(class_name, parent=superclass_node)

    # Now that the tree is built, we can set the indexes
    # This makes the tree read-only
    root.set_indexes()
    name_to_node_id = {node.name: root.node_to_id[node] for node in root.leaves}
    index_to_node_id = {
        i: name_to_node_id[name] for i, name in enumerate(train_data.classes)
    }

    # Render the hierarchy
    mo.Html(root.svg())
    return (
        HierarchicalSoftmaxLazyLinear,
        HierarchicalSoftmaxLoss,
        SoftmaxNode,
        index_to_node_id,
        root,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create DataLoaders with hierarchical labels""")
    return


@app.cell
def _(DataLoader, batch_size, index_to_node_id, test_data, torch, train_data):
    class HierarchicalDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, index_to_node_id):
            self.dataset = dataset
            self.index_to_node_id = index_to_node_id

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            return image, self.index_to_node_id[label]

        def __len__(self):
            return len(self.dataset)

    hierarchical_train_loader = DataLoader(HierarchicalDataset(train_data, index_to_node_id), batch_size=batch_size, shuffle=True)
    hierarchical_test_loader = DataLoader(HierarchicalDataset(test_data, index_to_node_id), batch_size=batch_size, shuffle=False)
    return hierarchical_test_loader, hierarchical_train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create the Hierarchical Image Classifier model""")
    return


@app.cell
def _(
    BasicImageClassifier,
    HierarchicalSoftmaxLazyLinear,
    HierarchicalSoftmaxLoss,
    SoftmaxNode,
    WideResNetBody,
    nn,
    root,
):
    from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric, LeafAccuracyTorchMetric

    class HierarchicalImageClassifier(BasicImageClassifier):
        # Just overriding the init - keep the rest of the code
        def __init__(self, root: SoftmaxNode, depth:int=28, width_factor:int=10, dropout_rate:float=0.3):
            super().__init__()
            self.model = nn.Sequential(
                WideResNetBody(depth=depth, width_factor=width_factor, dropout_rate=dropout_rate),
                HierarchicalSoftmaxLazyLinear(root=root)
            )
            self.loss_fn = HierarchicalSoftmaxLoss(root)
            self.metrics = [
                RankAccuracyTorchMetric(
                    root,
                    {1: "superclass_accuracy"},
                ),
                LeafAccuracyTorchMetric(root, name="class_accuracy"),
            ]
            self.root = root

    hierarchical_model = HierarchicalImageClassifier(root, depth=16, width_factor=8, dropout_rate=0.3)        
    hierarchical_model
    return (hierarchical_model,)


@app.cell
def _(
    CSVLogger,
    L,
    TQDMProgressBar,
    epochs,
    hierarchical_model,
    hierarchical_test_loader,
    hierarchical_train_loader,
):
    hierarchical_logger = CSVLogger(save_dir="lightning_logs", name="hierarchical_model")
    hierarchical_trainer = L.Trainer(max_epochs=epochs, accelerator="auto", enable_checkpointing=False, logger=hierarchical_logger, callbacks=[TQDMProgressBar(leave=True, refresh_rate=20)])
    hierarchical_trainer.fit(hierarchical_model, train_dataloaders=hierarchical_train_loader, val_dataloaders=hierarchical_test_loader)
    return (hierarchical_logger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot the validation results at both the superclass and the class levels""")
    return


@app.cell
def _(Path, go, hierarchical_logger, pd):
    hierarchical_df = pd.read_csv(Path(hierarchical_logger.log_dir) / "metrics.csv")
    hierarchical_df = hierarchical_df.dropna(subset=["val_LeafAccuracyTorchMetric"])
    hierarchical_fig = go.Figure()
    hierarchical_fig.add_trace(go.Scatter(x=hierarchical_df["epoch"], y=hierarchical_df["val_superclass_accuracy"], mode='lines', name='superclass'))
    hierarchical_fig.add_trace(go.Scatter(x=hierarchical_df["epoch"], y=hierarchical_df["val_LeafAccuracyTorchMetric"], mode='lines', name='class'))
    hierarchical_fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
    )
    hierarchical_fig.write_html(str(Path(hierarchical_logger.log_dir)/"accuracy.html"))
    hierarchical_fig
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
