---
title: Cifar
marimo-version: 0.13.7
width: full
header: |-
  # /// script
  # [tool.marimo.runtime]
  # auto_instantiate = false
  # ///
---

```python {.marimo hide_code="true"}
import marimo as mo
```

# CIFAR Demonstration

This notebook demonstrates how to use the `hierarchicalsoftmax` module to train a neural network on the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
<!---->
First, choose the hyperparameters.

```python {.marimo}
cifar_radio = mo.ui.radio(options=["10","100"], value=mo.cli_args().get("cifar") or "100", label="CIFAR Dataset")
batch_size_input = mo.ui.number(value=mo.cli_args().get("batch") or 32, label="Batch Size")
epochs_input = mo.ui.number(value=mo.cli_args().get("batch") or 10, label="Epochs")
mo.vstack([cifar_radio, epochs_input, batch_size_input])
```

```python {.marimo}
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
```

### Plot the first 10 images

```python {.marimo}
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
```

## Non-hierarchical model

First we create a basic non-hierarchical model as a baseline

```python {.marimo}
import torch
from torch import nn
from torchmetrics import Accuracy
import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNetBody(nn.Module):
    def __init__(self, depth=16, width_factor=8):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6

        k = width_factor
        self.in_planes = 16

        # Initial conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # 3 groups
        self.layer1 = self._make_layer(16*k, n, stride=1)
        self.layer2 = self._make_layer(32*k, n, stride=2)
        self.layer3 = self._make_layer(64*k, n, stride=2)

        self.bn = nn.BatchNorm2d(64*k)

    def _make_layer(self, out_planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, s))
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
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            WideResNetBody(),
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

basic_model = BasicImageClassifier()
basic_model
```

## Train the basic model

```python {.marimo}
from lightning.pytorch.loggers import CSVLogger

basic_logger = CSVLogger(save_dir="lightning_logs", name="basic_model")
basic_trainer = L.Trainer(max_epochs=epochs, accelerator="auto", enable_checkpointing=False, logger=basic_logger)
basic_trainer.fit(basic_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
```

```python {.marimo}
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
basic_fig.show()

```

## Hierarchical model

Let's now create a hierarchical model.
First we need to create a tree structure for the CIFAR dataset.

```python {.marimo}
from hierarchicalsoftmax import (
    SoftmaxNode,
    HierarchicalSoftmaxLazyLinear,
    HierarchicalSoftmaxLoss,
)
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric

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
```

### Create DataLoaders with hierarchical labels

```python {.marimo}
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
```

### Create the Hierarchical Image Classifier model

```python {.marimo}
class HierarchicalImageClassifier(BasicImageClassifier):
    # Just overriding the init - keep the rest of the code
    def __init__(self, root: SoftmaxNode):
        super().__init__()
        self.model = nn.Sequential(
            WideResNetBody(),
            HierarchicalSoftmaxLazyLinear(root=root)
        )
        self.loss_fn = HierarchicalSoftmaxLoss(root)
        self.metrics = [
            RankAccuracyTorchMetric(
                root,
                {1: "superclass_accuracy", 2: "class_accuracy"},
            ),
        ]
        self.root = root

hierarchical_model = HierarchicalImageClassifier(root)        
hierarchical_model
```

```python {.marimo}
hierarchical_logger = CSVLogger(save_dir="lightning_logs", name="hierarchical_model")
hierarchical_trainer = L.Trainer(max_epochs=epochs, accelerator="auto", enable_checkpointing=False, logger=hierarchical_logger)
hierarchical_trainer.fit(hierarchical_model, train_dataloaders=hierarchical_train_loader, val_dataloaders=hierarchical_test_loader)
```

### Plot the validation results at both the superclass and the class levels

```python {.marimo}
hierarchical_df = pd.read_csv(Path(hierarchical_logger.log_dir) / "metrics.csv")
hierarchical_df = hierarchical_df.dropna(subset=["val_class_accuracy"])
hierarchical_fig = go.Figure()
hierarchical_fig.add_trace(go.Scatter(x=hierarchical_df["epoch"], y=hierarchical_df["val_superclass_accuracy"], mode='lines', name='superclass'))
hierarchical_fig.add_trace(go.Scatter(x=hierarchical_df["epoch"], y=hierarchical_df["val_class_accuracy"], mode='lines', name='class'))
hierarchical_fig.update_layout(
    xaxis_title="Epochs",
    yaxis_title="Accuracy",
)
hierarchical_fig
```