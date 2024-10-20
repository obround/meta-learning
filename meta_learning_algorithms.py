"""
This file contains implementations of various meta-learning algorithms.

The implementations include:
- MAML: The original Model-Agnostic Meta-Learning algorithm
- MAML++: An improved version of MAML; only implemented per-step loss importance weighting from the paper
- Reptile: A simplified meta-learning algorithm that performs well with reduced computational cost
- MetaSGD: A meta-learning approach that learns a learning rate for each parameter

Author: Aditya Kumar (obround)

Note: This implementation contains some repetition and lacks abstraction across the model implementations.
      This code likely requires additional testing and optimization before production use.
"""

import torch
from torch import optim
from torch import nn
from torch.func import functional_call
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from copy import deepcopy
import higher


class MAML:
    """
    A class for implementing Model-Agnostic Meta-Learning (MAML) and MAML++ algorithms.

    Methods
    -------
    run_batch(batch, inner_steps, inner_opt, outer_opt, epochs, current_epoch, train=True):
        Executes a batch of tasks using either MAML or MAML++ depending on the configuration.

    run_batch_maml(batch, inner_steps, inner_opt, outer_opt, train):
        Runs the MAML algorithm by optimizing task-specific parameters and updating meta-parameters based on the query loss.

    run_batch_mpp(batch, inner_steps, inner_opt, outer_opt, epochs, current_epoch, train):
        Executes the MAML++ algorithm by applying loss importance weighting over multiple inner-loop steps.

    validate(val_loader, inner_steps=5):
        Computes the average loss over the validation set without updating meta-parameters.

    get_per_step_loss_importance_vector(epochs, current_epoch, inner_steps):
        Calculates the loss weighting vector for each inner-loop step during MAML++ training, applying decay to intermediate steps.

    train(train_loader, val_loader=None, epochs=20, inner_steps=5, early_stopper=None, verbose=True):
        Trains the model across multiple epochs using meta-learning, evaluates it on validation tasks, and supports early stopping.
    """

    def __init__(
            self,
            model,
            loss,
            maml_plus_plus=False,
            inner_lr=1e-3,
            meta_lr=1e-3,
            clipping=1.0,
            device="cpu"
    ):
        """
        Initializes the MAML algorithm trainer.

        :param model: The neural network model to be optimized.
        :type model: torch.nn.Module
        :param loss: The loss function used to compute training loss.
        :type loss: torch.nn.Module
        :param inner_lr: The task-specific inner-loop learning rate, defaults to 1e-3.
        :type inner_lr: float, optional
        :param meta_lr: The meta learning rate for the meta-optimizer, defaults to 1e-3.
        :type meta_lr: float, optional
        :param clipping: Max norm for gradient clipping to avoid gradient issues, defaults to 1.0.
        :type clipping: float, optional
        :param device: Device to run the model on ("cpu", "cuda", or "mps"), defaults to "cpu".
        :type device: str, optional
        """
        self.maml_plus_plus = maml_plus_plus
        self.device = device
        self.clipping = clipping
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.inner_lr = inner_lr  # Inner loop learning rate
        self.meta_lr = meta_lr  # Outer loop learning rate

    def run_batch_mpp(self, batch, inner_steps, inner_opt, outer_opt, epochs, current_epoch, train):
        """
        Executes a batch of tasks using the MAML++ algorithm, which applies a per-step loss importance weighting during
        the inner loop of meta-learning.

        :param batch: A batch of tasks where each task consists of support and query sets.
        :type batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
        :param inner_steps: Number of inner-loop optimization steps for each task.
        :type inner_steps: int
        :param inner_opt: Optimizer for the inner-loop updates (task-specific updates).
        :type inner_opt: torch.optim.Optimizer
        :param outer_opt: Optimizer for the outer-loop updates (meta-optimization).
        :type outer_opt: torch.optim.Optimizer
        :param epochs: Total number of training epochs, used to adjust step importance weights.
        :type epochs: int
        :param current_epoch: The current epoch number, used to adjust the importance weights of each inner step.
        :type current_epoch: int
        :param train: Boolean flag indicating whether to perform training (backpropagation and optimizer step) or only forward pass.
        :type train: bool
        :return: The meta-loss for the current batch.
        :rtype: float
        """
        outer_opt.zero_grad()
        task_losses = []

        for support, query in batch:
            support_features, support_targets = support
            query_features, query_targets = query

            with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # Inner loop
                importance_vector = self.get_per_step_loss_importance_vector(epochs, current_epoch, inner_steps)
                for step in range(inner_steps):
                    fmodel.train()
                    support_predictions = fmodel(support_features)
                    inner_loss = self.loss(support_predictions, support_targets)
                    diffopt.step(inner_loss)

                    fmodel.eval()
                    query_predictions = fmodel(query_features)
                    task_loss = self.loss(query_predictions, query_targets)
                    task_losses.append(task_loss * importance_vector[step])

        meta_loss = sum(task_losses)
        if train:
            meta_loss.backward()
            outer_opt.step()

        return meta_loss.item()

    def run_batch_maml(self, batch, inner_steps, inner_opt, outer_opt, train):
        """
        Executes a batch of tasks using the original MAML algorithm, which performs a fixed number of inner-loop updates
        for each task without applying any additional loss weighting between steps.

        :param batch: A batch of tasks where each task consists of support and query sets.
        :type batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
        :param inner_steps: Number of inner-loop optimization steps for each task.
        :type inner_steps: int
        :param inner_opt: Optimizer for the inner-loop updates (task-specific updates).
        :type inner_opt: torch.optim.Optimizer
        :param outer_opt: Optimizer for the outer-loop updates (meta-optimization).
        :type outer_opt: torch.optim.Optimizer
        :param train: Boolean flag indicating whether to perform training (backpropagation and optimizer step) or only forward pass.
        :type train: bool
        :return: The meta-loss for the current batch.
        :rtype: float
        """
        outer_opt.zero_grad()
        task_losses = []

        for support, query in batch:
            support_features, support_targets = support
            query_features, query_targets = query

            with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # Inner loop
                fmodel.train()
                for step in range(inner_steps):
                    support_predictions = fmodel(support_features)
                    inner_loss = self.loss(support_predictions, support_targets)
                    diffopt.step(inner_loss)

                fmodel.eval()
                query_predictions = fmodel(query_features)
                task_loss = self.loss(query_predictions, query_targets)
                task_losses.append(task_loss)

        meta_loss = sum(task_losses) / len(batch)
        if train:
            meta_loss.backward()
            outer_opt.step()

        return meta_loss.item()

    def run_batch(self, batch, inner_steps, inner_opt, outer_opt, epochs, current_epoch, train=True):
        """
        Executes a batch of tasks by selecting either the MAML or MAML++ algorithm.

        :param batch: A batch of tasks where each task consists of support and query sets.
        :type batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
        :param inner_steps: Number of inner-loop optimization steps for each task.
        :type inner_steps: int
        :param inner_opt: Optimizer for the inner-loop updates (task-specific updates).
        :type inner_opt: torch.optim.Optimizer
        :param outer_opt: Optimizer for the outer-loop updates (meta-optimization).
        :type outer_opt: torch.optim.Optimizer
        :param epochs: Total number of training epochs, used to adjust step importance weights in MAML++.
        :type epochs: int
        :param current_epoch: The current epoch number, used to adjust the importance weights of each inner step in MAML++.
        :type current_epoch: int
        :param train: Boolean flag indicating whether to perform training (backpropagation and optimizer step) or only forward pass.
        :type train: bool
        :return: The meta-loss for the current batch.
        :rtype: float
        """
        if self.maml_plus_plus:
            return self.run_batch_mpp(
                batch,
                inner_steps,
                inner_opt,
                outer_opt,
                epochs,
                current_epoch,
                train
            )
        return self.run_batch_maml(
            batch,
            inner_steps,
            inner_opt,
            outer_opt,
            train
        )

    def validate(self, val_loader, inner_steps=5):
        """
        Validates the model on the entire validation set by computing the average loss.

        :param val_loader: DataLoader for the validation set.
        :type val_loader: torch.utils.data.DataLoader
        :param inner_steps: Number of inner training steps per task, defaults to 5.
        :type inner_steps: int, optional
        :return: The average loss over the validation set.
        :rtype: float
        """
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        outer_opt = optim.Adam(self.model.parameters(), lr=self.meta_lr)

        return sum(
            self.run_batch(
                batch=batch,
                inner_steps=inner_steps,
                inner_opt=inner_opt,
                outer_opt=outer_opt,
                epochs=1,
                current_epoch=1,
                train=False
            )
            for batch in val_loader
        ) / len(val_loader)

    def get_per_step_loss_importance_vector(self, epochs, current_epoch, inner_steps):
        """
        Computes a vector that assigns varying importance to the loss at each step of the inner loop during meta-learning.

        The importance of each inner-loop step is weighted such that earlier steps are given progressively lower weights
        as training progresses through epochs, and the final step is emphasized more as training reaches later epochs.

        This function was taken from the official How To Train Your MAML source code.

        :param epochs: Total number of training epochs.
        :type epochs: int
        :param current_epoch: The current epoch number (used for adjusting the weights over time).
        :type current_epoch: int
        :param inner_steps: Number of inner-loop optimization steps per task.
        :type inner_steps: int
        :return: A tensor of shape `(inner_steps,)` containing the loss importance weights for each inner step.
        :rtype: torch.Tensor
        """
        loss_weights = torch.ones(size=(inner_steps,)) * (1.0 / inner_steps)
        decay_rate = 1.0 / inner_steps / epochs
        min_value_for_non_final_losses = 0.03 / inner_steps

        for i in range(len(loss_weights) - 1):
            curr_value = torch.max(loss_weights[i] - (current_epoch * decay_rate),
                                   torch.tensor(min_value_for_non_final_losses))
            loss_weights[i] = curr_value

        curr_value = torch.min(
            loss_weights[-1] + (current_epoch * (inner_steps - 1) * decay_rate),
            torch.tensor(1.0 - ((inner_steps - 1) * min_value_for_non_final_losses))
        )
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(self.device)

        return loss_weights

    def train(
            self,
            train_loader,
            val_loader=None,
            epochs=20,
            inner_steps=5,
            early_stopper=None,
            verbose=True
    ):
        """
        Trains the model over multiple epochs, evaluates on the validation set, and applies
        early stopping if specified.

        :param train_loader: DataLoader for the training set.
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: DataLoader for the validation set.
        :type val_loader: torch.utils.data.DataLoader, optional
        :param epochs: Number of training epochs, defaults to 50.
        :type epochs: int, optional
        :param inner_steps: Number of inner training steps per task, defaults to 5.
        :type inner_steps: int, optional
        :param early_stopper: Callback function for early stopping, defaults to None.
        :type early_stopper: utils.EarlyStopper, optional
        :param verbose: If True, displays training progress, defaults to True.
        :type verbose: bool, optional
        """
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        outer_opt = optim.Adam(self.model.parameters(), lr=self.meta_lr)

        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", disable=not verbose)
            train_batch_losses = []
            val_loss = 0.
            for i, batch in enumerate(pbar, 1):
                train_loss = self.run_batch(
                    batch=batch,
                    inner_steps=inner_steps,
                    inner_opt=inner_opt,
                    outer_opt=outer_opt,
                    epochs=epochs,
                    current_epoch=epoch,
                    train=True
                )
                train_batch_losses.append(train_loss)
                if i == len(train_loader) and val_loader is not None:
                    val_loss = self.validate(val_loader)
                    pbar.set_postfix({
                        "loss": sum(train_batch_losses) / len(train_batch_losses),
                        "val_loss": val_loss
                    })
                else:
                    pbar.set_postfix({"loss": sum(train_batch_losses) / len(train_batch_losses)})
            if early_stopper and early_stopper.early_stop(val_loss, model=self.model):
                break


class Reptile:
    """
    Implements Reptile, a meta-learning algorithm cheaper than MAML with on-par performance.

    Methods
    -------
    run_batch(batch, inner_steps, train=True):
        Trains/Validates the model on a batch of tasks by computing the meta-loss, and, if training,
        updating the model parameters.

    validate(val_loader, inner_steps):
        Validates the model on the entire validation set by computing the average loss.

    train(train_loader, val_loader, epochs=50, inner_step=5, early_stopper=None, verbose=True):
        Trains the model over multiple epochs, evaluates on the validation set, and applies
        early stopping if specified.
    """

    def __init__(self, model, loss, inner_lr=1e-3, meta_lr=1e-3, clipping=1.0, device="cpu"):
        """
        Initializes the Reptile algorithm trainer.

        :param model: The neural network model to be optimized.
        :type model: torch.nn.Module
        :param loss: The loss function used to compute training loss.
        :type loss: torch.nn.Module
        :param inner_lr: The task-specific inner-loop learning rate, defaults to 1e-3.
        :type inner_lr: float, optional
        :param meta_lr: The meta learning rate for the meta-optimizer, defaults to 1e-3.
        :type meta_lr: float, optional
        :param clipping: Max norm for gradient clipping to avoid gradient issues, defaults to 1.0.
        :type clipping: float, optional
        :param device: Device to run the model on ("cpu", "cuda", or "mps"), defaults to "cpu".
        :type device: str, optional
        """
        self.device = device
        self.clipping = clipping
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.inner_lr = inner_lr  # Inner loop learning rate
        self.meta_lr = meta_lr  # Outer loop learning rate

    def run_batch(self, batch, inner_steps, train=True):
        """
        Trains/Validates the model on a batch of tasks by computing the meta-loss, and, if training,
        updating the model parameters.

        :param batch: A batch containing support and query sets for multiple tasks.
        :type batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
        :param inner_steps: Number of inner training steps per task.
        :type inner_steps: int
        :param train: Whether to train the model parameters on the task losses
        :return: The meta-loss for the batch after the meta-update step.
        :rtype: float
        """
        meta_losses = []
        meta_grad = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        for task in batch:
            task_features, task_targets = task

            # Clone the model for each task
            task_model = deepcopy(self.model)
            inner_opt = optim.SGD(task_model.parameters(), lr=self.inner_lr)

            # Inner loop (task-specific adaptation)
            inner_loss = 0.
            for _ in range(inner_steps):
                task_model.train()
                task_predictions = task_model(task_features)
                inner_loss = self.loss(task_predictions, task_targets)

                inner_opt.zero_grad()
                inner_loss.backward()
                if self.clipping is not None:
                    torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=self.clipping)
                inner_opt.step()

            meta_losses.append(inner_loss.item() / len(task))

            if train:
                with torch.no_grad():
                    for (name, param), (name_task, param_task) in zip(self.model.named_parameters(),
                                                                      task_model.named_parameters()):
                        meta_grad[name].add_(param_task - param)

        if train:
            # Apply meta-update
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.add_(self.meta_lr * meta_grad[name] / len(batch))

        return sum(meta_losses) / len(meta_losses)

    def validate(self, val_loader, inner_steps=5):
        """
        Validates the model on the entire validation set by computing the average loss.

        :param val_loader: DataLoader for the validation set.
        :type val_loader: torch.utils.data.DataLoader
        :param inner_steps: Number of inner training steps per task, defaults to 5.
        :type inner_steps: int, optional
        :return: The average loss over the validation set.
        :rtype: float
        """
        return sum(
            self.run_batch(batch=batch, inner_steps=inner_steps, train=False)
            for batch in val_loader
        ) / len(val_loader)

    def train(
            self,
            train_loader,
            val_loader=None,
            epochs=20,
            inner_steps=5,
            early_stopper=None,
            verbose=True
    ):
        """
        Trains the model over multiple epochs, evaluates on the validation set, and applies
        early stopping if specified.

        :param train_loader: DataLoader for the training set.
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: DataLoader for the validation set.
        :type val_loader: torch.utils.data.DataLoader, optional
        :param epochs: Number of training epochs, defaults to 50.
        :type epochs: int, optional
        :param inner_steps: Number of inner training steps per task, defaults to 5.
        :type inner_steps: int, optional
        :param early_stopper: Callback function for early stopping, defaults to None.
        :type early_stopper: utils.EarlyStopper, optional
        :param verbose: If True, displays training progress, defaults to True.
        :type verbose: bool, optional
        """
        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", disable=not verbose)
            train_batch_losses = []
            val_loss = 0.
            for i, batch in enumerate(pbar, 1):
                train_loss = self.run_batch(batch=batch, inner_steps=inner_steps, train=True)
                train_batch_losses.append(train_loss)
                if i == len(train_loader) and val_loader is not None:
                    val_loss = self.validate(val_loader)
                    pbar.set_postfix({
                        "loss": sum(train_batch_losses) / len(train_batch_losses),
                        "val_loss": val_loss
                    })
                else:
                    pbar.set_postfix({"loss": sum(train_batch_losses) / len(train_batch_losses)})
            if early_stopper and early_stopper.early_stop(val_loss, model=self.model):
                break


class MetaSGD:
    """
    Implements MetaSGD, a meta-learning algorithm where the task-specific learning rate is learned
    along with the model parameters. This algorithm adapts the model parameters to each task's support set
    and optimizes for the performance on the query set, updating the learning rate using a meta-optimizer.

    Methods
    -------
    loss_on_task(support, query):
        Computes the loss for a single task based on the support and query sets.

    train_batch(batch):
        Trains the model on a batch of tasks by performing meta-optimization on the meta-loss.

    validate_batch(batch):
        Evaluates the model on a batch of tasks without updating the model parameters.

    validate(val_loader):
        Computes the average loss over the entire validation set.

    train(train_loader, val_loader, epochs=50, early_stopper=None, verbose=True):
        Trains the model over multiple epochs, evaluates on the validation set, and applies early stopping.
    """

    def __init__(self, model, loss, inner_lr=1e-3, meta_lr=5e-4, clipping=1.0, device="cpu", ):
        """
        Initializes the MetaSGD algorithm by setting up the model, task-specific learning rates,
        and the meta-optimizer.

        :param model: The neural network model to be optimized.
        :type model: torch.nn.Module
        :param loss: The loss function used to compute training loss.
        :type loss: torch.nn.Module
        :param inner_lr: The initial learning rate for task-specific learning, defaults to 1e-3.
        :type inner_lr: float, optional
        :param meta_lr: The meta learning rate for the meta-optimizer, defaults to 5e-4.
        :type meta_lr: float, optional
        :param clipping: Max norm for gradient clipping to avoid gradients issues, defaults to 1.0.
        :type clipping: float, optional
        :param device: Device to run the model on ("cpu", "cuda", or "mps"), defaults to "cpu".
        :type device: str, optional
        """
        self.device = device
        self.clipping = clipping
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.task_lr = {
            name: nn.Parameter(torch.ones_like(param) * inner_lr, requires_grad=True)
            for name, param in self.model.named_parameters()
        }
        self.meta_optimizer = optim.Adam(list(self.model.parameters()) + list(self.task_lr.values()), lr=meta_lr)

    def loss_on_task(self, support, query):
        """
        Computes the task-specific loss by adapting the model to the support set and
        evaluating on the query set.

        :param support: A tuple containing support features and support targets.
        :type support: Tuple[torch.Tensor, torch.Tensor]
        :param query: A tuple containing query features and query targets.
        :type query: Tuple[torch.Tensor, torch.Tensor]
        :return: The loss on the query set after adapting the model on the support set.
        :rtype: torch.Tensor
        """
        support_features, support_targets = support
        query_features, query_targets = query

        # Clone model parameters
        adapted_state_dict = OrderedDict({
            name: param.clone()
            for name, param in self.model.named_parameters()
        })

        # Forward pass on support set
        support_predictions = self.model(support_features)
        support_loss = self.loss(support_predictions, support_targets) / len(support_targets)

        # Backpropagate to obtain gradients
        grads = torch.autograd.grad(support_loss, list(self.model.parameters()), create_graph=True)
        if self.clipping is not None:
            nn.utils.clip_grad_norm_(grads, max_norm=self.clipping)

        # Update model parameters with task-specific learning rate
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            adapted_state_dict[name] = param - self.task_lr[name] * grad

        # Evaluate loss on the query set
        query_predictions = functional_call(self.model, adapted_state_dict, query_features)
        query_loss = self.loss(query_predictions, query_targets) / len(query_targets)

        return query_loss

    def train_batch(self, batch):
        """
        Trains the model on a batch of tasks by computing the meta-loss and updating the model
        parameters and learning rates.

        :param batch: A batch containing support and query sets for multiple tasks.
        :type batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
        :return: The meta-loss for the batch after the meta-update step.
        :rtype: float
        """
        meta_loss = sum(self.loss_on_task(support, query) for support, query in batch) / len(batch)

        # Perform meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        if self.clipping is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clipping)
            nn.utils.clip_grad_norm_(self.task_lr.values(), max_norm=self.clipping)
        self.meta_optimizer.step()

        return meta_loss.item()

    def validate_batch(self, batch):
        """
        Validates the model on a batch of tasks by computing the average loss without updating the model.

        :param batch: A batch containing support and query sets for multiple tasks.
        :type batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
        :return: The average loss for the batch.
        :rtype: float
        """
        return sum(self.loss_on_task(support, query) for support, query in batch).item() / len(batch)

    def validate(self, val_loader):
        """
        Validates the model on the entire validation set by computing the average loss.

        :param val_loader: DataLoader for the validation set.
        :type val_loader: torch.utils.data.DataLoader
        :return: The average loss over the validation set.
        :rtype: float
        """
        return sum(self.validate_batch(batch) for batch in val_loader) / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=50, early_stopper=None, verbose=True):
        """
        Trains the model over multiple epochs, evaluates on the validation set, and applies
        early stopping if specified.

        :param train_loader: DataLoader for the training set.
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: DataLoader for the validation set.
        :type val_loader: torch.utils.data.DataLoader, optional
        :param epochs: Number of training epochs, defaults to 50.
        :type epochs: int, optional
        :param early_stopper: Callback function for early stopping, defaults to None.
        :type early_stopper: utils.EarlyStopper, optional
        :param verbose: If True, displays training progress, defaults to True.
        :type verbose: bool, optional
        """
        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", disable=not verbose)
            train_batch_losses = []
            val_loss = 0.
            for i, batch in enumerate(pbar, 1):
                train_loss = self.train_batch(batch)
                train_batch_losses.append(train_loss)
                if i == len(train_loader) and val_loader is not None:
                    val_loss = self.validate(val_loader)
                    pbar.set_postfix({
                        "loss": sum(train_batch_losses) / len(train_batch_losses),
                        "val_loss": val_loss
                    })
                else:
                    pbar.set_postfix({"loss": sum(train_batch_losses) / len(train_batch_losses)})
            if early_stopper and early_stopper.early_stop(val_loss, model=self.model):
                break


class EarlyStopper:
    """
    A class to implement early stopping during model training.
    The best model weights are saved when the validation loss improves.

    Modified version of StackOverflow user isle_of_gods' code.

    Methods
    -------
    early_stop(validation_loss, model):
        Checks if training should be stopped based on the current validation loss.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        Initializes the EarlyStopper instance with specified patience and minimum delta.

        :param patience: Number of epochs to wait for improvement before stopping training, defaults to 1.
        :type patience: int, optional
        :param min_delta: Minimum change in validation loss to be considered an improvement, defaults to 0.
        :type min_delta: float, optional
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss, model, file="maml-demo.pt"):
        """
        Evaluates the current validation loss to determine if training should be stopped.

        If the validation loss has improved, the model's state is saved and the counter is reset.
        If the validation loss has not improved by at least min_delta, the counter is incremented.
        If the counter exceeds patience, early stopping is triggered.

        :param validation_loss: The current validation loss to evaluate.
        :type validation_loss: float
        :param model: The model whose state is to be saved if the validation loss improves.
        :type model: torch.nn.Module
        :param file: The file to save the best model weights to.
        :type file: str
        :return: True if training should be stopped, False otherwise.
        :rtype: bool
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), file)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
