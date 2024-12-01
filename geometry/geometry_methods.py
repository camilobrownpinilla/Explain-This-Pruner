from pyhessian import hessian
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def compute_gradient_statistics(
    model,
    tokenizer,
    dataset,
    loss_fn,
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    params:
        model: Hugging Face model to evaluate.
        tokenizer: Tokenizer associated with the Hugging Face model.
        dataset: Dataset to use, in Hugging Face Dataset format.
        loss_fn: loss Loss function (should accept logits and labels).
        batch_size: Number of samples per batch of computation (default is 16).
        device: Device to perform computations on, either "cuda" for GPU or "cpu" (default is "cuda" if available).

    returns:
        stats: dictionary containing the following gradient magnitude statistics:
            median, mean, variance, standard deviation, max, min, skewness, and kurtosis.
    """
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    gradient_magnitudes = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs = tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = batch["label"].to(device)

        # Enable gradient computation for input embeddings
        inputs["input_ids"].requires_grad = True

        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()  

        # Fetch gradients of the input embeddings
        grads = inputs["input_ids"].grad
        grad_magnitude = torch.norm(grads, dim=-1)  # Compute L2 norm of the gradients
        gradient_magnitudes.append(grad_magnitude.cpu().detach().numpy())

    gradient_magnitudes = np.concatenate(gradient_magnitudes)
    stats = {
        "median_gradient_magnitude": np.median(gradient_magnitudes),
        "average_gradient_magnitude": np.mean(gradient_magnitudes),
        "variance_gradient_magnitude": np.var(gradient_magnitudes),
        "std_gradient_magnitude": np.std(gradient_magnitudes),
        "max_gradient_magnitude": np.max(gradient_magnitudes),
        "min_gradient_magnitude": np.min(gradient_magnitudes),
        "skewness_gradient_magnitude": skew(gradient_magnitudes),
        "kurtosis_gradient_magnitude": kurtosis(gradient_magnitudes),
    }

    return stats




def compute_hessian_statistics(
    model,
    tokenizer,
    dataset,
    loss_fn,
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    params:
        model: Hugging Face model to evaluate.
        tokenizer: Tokenizer corresponding to the Hugging Face model.
        dataset: Dataset in a format compatible with Hugging Face AND PyTorch.
        loss_fn: PyTorch loss function.
        batch_size: Number of samples per batch of computation (default is 16).
        device: Device to perform computations on, either "cuda" for GPU or "cpu" (default is "cuda" if available).

    returns:
        stats: dictionary containing the following Hessian eigenvalue statistics:
            median, mean, variance, standard deviation, max, min, skewness, and kurtosis.
    """
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    hessian_eigenvalues = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs = tokenizer(
            batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = batch['label'].to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        # Compute Hessian eigenvalues using PyHessian
        hessian_comp = hessian(model, loss, data=(inputs, labels))
        eigenvalues, _ = hessian_comp.eigenvalues()
        eigenvalues = torch.tensor(eigenvalues).abs()
        hessian_eigenvalues.append(eigenvalues.cpu().numpy())

    hessian_eigenvalues = np.concatenate(hessian_eigenvalues)
    stats = {
        "median_hessian_eigenvalue": np.median(hessian_eigenvalues),
        "average_hessian_eigenvalue": np.mean(hessian_eigenvalues),
        "variance_hessian_eigenvalue": np.var(hessian_eigenvalues),
        "std_hessian_eigenvalue": np.std(hessian_eigenvalues),
        "max_hessian_eigenvalue": np.max(hessian_eigenvalues),
        "min_hessian_eigenvalue": np.min(hessian_eigenvalues),
        "skewness_hessian_eigenvalue": skew(hessian_eigenvalues),
        "kurtosis_hessian_eigenvalue": kurtosis(hessian_eigenvalues),
    }

    return stats