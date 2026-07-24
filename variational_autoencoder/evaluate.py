import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from pytorch_msssim import SSIM

# Import the VAE model
from model import VAE
from dataset import MVTecDataset

# Define image dimensions and hyperparameters (must match training)
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
LOSS_FUNCTION_TYPE = 'l2' # 'l2' for MSE, 'ssim' for SSIM-based loss
LATENT_DIM = 128

# Data specific configuration
target_object = 'wood' # Must match training object
detect_category = '' # default: empty string, '' for all anomalies, or specify an anomaly type

# Define directory for saving models
MODEL_SAVE_DIR = './saved_models/' + target_object
TEST_EPOCH = 50 # Must match the epoch of the saved model you want to load

# Define postfix for model filename
model_filename = f'vae_{target_object}_{LOSS_FUNCTION_TYPE}_epoch{TEST_EPOCH}'
if detect_category:
    model_filename_prefix = f'{model_filename}_{detect_category}'
else:
    model_filename_prefix = model_filename

# Set up logging to both console and a log file alongside the saved models
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
log_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_evaluate.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Ensure CUDA is available for GPU inference, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define transformations (must match training)
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(), # Converts PIL Image to PyTorch Tensor (H*W*C to C*H*W) and normalizes to [0, 1]
])

# Helper function to calculate reconstruction errors
# NOTE: model.eval() makes VAE.forward() use the deterministic latent (z = mu,
# no sampling), so the reconstruction error here is directly comparable to the
# plain autoencoder's reconstruction error.
def calculate_reconstruction_errors(dataloader, model, device, loss_type='l2'):
    model.eval()
    reconstruction_errors = []
    if dataloader is None:
        return np.array([])

    if loss_type == 'l2':
        error_criterion = nn.MSELoss(reduction='none')
    elif loss_type == 'ssim':
        error_criterion = SSIM(data_range=1.0, size_average=False, channel=1)
    else:
        raise ValueError(f"Unknown loss_type for error calculation: {loss_type}")

    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            outputs, _, _ = model(inputs)

            if loss_type == 'l2':
                error = torch.mean(error_criterion(outputs, inputs), dim=[1, 2, 3])
            elif loss_type == 'ssim':
                ssim_values = error_criterion(outputs, inputs)
                error = 1 - ssim_values
            reconstruction_errors.extend(error.cpu().numpy())
    return np.array(reconstruction_errors)

# Helper function to get original images, reconstructed images, and their errors from a dataloader
def get_reconstructions_and_errors_for_dataset(dataloader, model, device, loss_type='l2'):
    model.eval()
    original_images_list = []
    reconstructed_images_list = []
    reconstruction_errors = []

    if dataloader is None:
        return [], [], np.array([])

    if loss_type == 'l2':
        error_criterion = nn.MSELoss(reduction='none')
    elif loss_type == 'ssim':
        error_criterion = SSIM(data_range=1.0, size_average=False, channel=1)
    else:
        raise ValueError(f"Unknown loss_type for error calculation: {loss_type}")

    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            outputs, _, _ = model(inputs)

            if loss_type == 'l2':
                error = torch.mean(error_criterion(outputs, inputs), dim=[1, 2, 3])
            elif loss_type == 'ssim':
                ssim_values = error_criterion(outputs, inputs)
                error = 1 - ssim_values

            original_images_list.extend([img.cpu().squeeze().numpy() for img in inputs])
            reconstructed_images_list.extend([img.cpu().squeeze().numpy() for img in outputs])
            reconstruction_errors.extend(error.cpu().numpy())

    return original_images_list, reconstructed_images_list, np.array(reconstruction_errors)

# Helper function to visualize a list of original, reconstructed, and error images
def visualize_filtered_reconstructions(original_imgs, reconstructed_imgs, errors, save_path, num_images=5, title=""):
    if not original_imgs:
        logger.info(f"No images to visualize for {title}.")
        return

    num_to_display = min(num_images, len(original_imgs))
    plt.figure(figsize=(num_to_display * 2.5, 7))
    for i in range(num_to_display):
        # Original Image
        plt.subplot(3, num_to_display, i + 1)
        plt.imshow(original_imgs[i], cmap='gray')
        plt.title(f"Original")
        plt.axis('off')

        # Reconstructed Image
        plt.subplot(3, num_to_display, num_to_display + i + 1)
        plt.imshow(reconstructed_imgs[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        # Error Map
        plt.subplot(3, num_to_display, 2 * num_to_display + i + 1)
        img_error_map = np.abs(original_imgs[i] - reconstructed_imgs[i])
        plt.imshow(img_error_map, cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Error ({errors[i]:.4f})")
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {save_path}")

# Load test 'good' images
base_dir = '/content/mvtec/' + target_object
test_good_image_dir = os.path.join(base_dir, 'test/good')
test_good_dataset = MVTecDataset(root_dir=test_good_image_dir, transform=transform)
test_good_loader = DataLoader(test_good_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Auto-generate anomaly_subdirs_to_display (assuming 'test' folder structure)
test_dir = os.path.join(base_dir, 'test')
anomaly_subdirs_to_display = [
    d for d in os.listdir(test_dir)
    if os.path.isdir(os.path.join(test_dir, d)) and d != 'good' if detect_category == '' or d == detect_category
]

# Load each anomaly (defect) type as its own dataset/loader, keyed by subtype name,
# instead of pooling them into one anonymous dataset. Per-type identity is kept
# through error calculation and is only pooled where an aggregate statistic (ROC,
# confusion matrix, etc.) genuinely needs all anomalies combined.
anomaly_loaders_by_type = {}
for subdir in anomaly_subdirs_to_display:
    current_anomaly_dir = os.path.join(base_dir, 'test', subdir)
    if os.path.exists(current_anomaly_dir):
        anomaly_dataset = MVTecDataset(root_dir=current_anomaly_dir, transform=transform)
        anomaly_loaders_by_type[subdir] = DataLoader(anomaly_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        logger.warning(f"Anomaly directory not found: {current_anomaly_dir}")


dict_type_num = {}
logger.info(f"Number of test good images: {len(test_good_dataset)}")
if anomaly_loaders_by_type:
    for subtype, loader in anomaly_loaders_by_type.items():
        logger.info(f"Number of test '{subtype}' anomaly images: {len(loader.dataset)}")
        dict_type_num[subtype] = len(loader.dataset)
    logger.info(f"Number of combined test anomaly images: {sum(len(loader.dataset) for loader in anomaly_loaders_by_type.values())}")
else:
    logger.info("No anomaly datasets were loaded.")

# Instantiate the model and move to device
model = VAE(latent_dim=LATENT_DIM).to(device)

# Load the trained model weights
# You might need to adjust the epoch number here if you want to load a different saved model
model_load_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename}.pth')

if os.path.exists(model_load_path):
    model.load_state_dict(torch.load(model_load_path))
    logger.info(f"Loaded model from {model_load_path}")
else:
    logger.error(f"Model not found at {model_load_path}. Please ensure training was successful.")
    exit()

# Calculate errors for good test images
good_errors = calculate_reconstruction_errors(test_good_loader, model, device, loss_type=LOSS_FUNCTION_TYPE)

# Calculate errors per anomaly type, then pool them for the aggregate statistics below
anomaly_errors_by_type = {
    subtype: calculate_reconstruction_errors(loader, model, device, loss_type=LOSS_FUNCTION_TYPE)
    for subtype, loader in anomaly_loaders_by_type.items()
}
anomaly_errors = np.concatenate(list(anomaly_errors_by_type.values())) if anomaly_errors_by_type else np.array([])

logger.info(f"Mean reconstruction error for good images: {np.mean(good_errors):.4f}")
for subtype, errors in anomaly_errors_by_type.items():
    logger.info(f"Mean reconstruction error for '{subtype}' anomaly images: {np.mean(errors):.4f} (n={len(errors)})")
logger.info(f"Mean reconstruction error for anomaly images (all types combined): {np.mean(anomaly_errors):.4f}")

# Visualize Reconstruction Error Distribution
plt.figure(figsize=(10, 6))
plt.hist(good_errors, bins=50, alpha=0.7, label='Good Images (Reconstruction Error)', color='blue')
plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly Images (Reconstruction Error)', color='red')
plt.title('Distribution of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
error_distribution_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_reconstruction_error_distribution.png')
plt.savefig(error_distribution_path)
plt.close()
logger.info(f"Saved plot to {error_distribution_path}")

# Determine Optimal Anomaly Threshold (ROC Curve)
y_true = np.concatenate((np.zeros(len(good_errors)), np.ones(len(anomaly_errors))))
y_scores = np.concatenate((good_errors, anomaly_errors))

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
roc_curve_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_roc_curve.png')
plt.savefig(roc_curve_path)
plt.close()
logger.info(f"Saved plot to {roc_curve_path}")

logger.info(f"Area Under the Curve (AUC): {roc_auc:.4f}")

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
logger.info(f"Optimal anomaly threshold: {optimal_threshold:.4f}")

# Visualize Confusion Matrix
y_pred = (y_scores > optimal_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
confusion_matrix_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.close()
logger.info(f"Saved plot to {confusion_matrix_path}")

logger.info("Classification Report:\n" + classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))


# 1. Get original, reconstructed images and errors per anomaly type, then pool them
# (keeping a parallel subtype label per image so false negatives can be broken down by type)
anomaly_originals = []
anomaly_reconstructions = []
anomaly_errors_full_list = []
anomaly_subtype_labels = []
for subtype, loader in anomaly_loaders_by_type.items():
    originals, reconstructions, errors_full = get_reconstructions_and_errors_for_dataset(
        loader, model, device, loss_type=LOSS_FUNCTION_TYPE
    )
    anomaly_originals.extend(originals)
    anomaly_reconstructions.extend(reconstructions)
    anomaly_errors_full_list.extend(errors_full)
    anomaly_subtype_labels.extend([subtype] * len(errors_full))
anomaly_errors_full = np.array(anomaly_errors_full_list)

# 2. Identify the indices where anomaly errors are <= optimal_threshold
# These are the anomaly images that were incorrectly classified as 'good' (false negatives)
false_negatives_indices = np.where(anomaly_errors_full <= optimal_threshold)[0]

# 3. Filter the images and errors based on these indices
detected_as_good_originals = [anomaly_originals[i] for i in false_negatives_indices]
detected_as_good_reconstructions = [anomaly_reconstructions[i] for i in false_negatives_indices]
detected_as_good_errors = [anomaly_errors_full[i] for i in false_negatives_indices]
detected_as_good_subtypes = [anomaly_subtype_labels[i] for i in false_negatives_indices]

logger.info(f"Found {len(detected_as_good_originals)} anomaly images detected as 'good' (error <= {optimal_threshold:.4f}).")
for subtype in sorted(set(detected_as_good_subtypes)):
    logger.info(f"  - {detected_as_good_subtypes.count(subtype)} from '{subtype}' (total {dict_type_num.get(subtype, 0)} images in test set)")

# 4. Visualize these filtered anomaly images (false negatives)
visualize_filtered_reconstructions(
    detected_as_good_originals,
    detected_as_good_reconstructions,
    detected_as_good_errors,
    save_path=os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_false_negatives.png'),
    num_images=5, # Display up to 5 such images
    title="Anomaly Images Detected as 'Good' (False Negatives)"
)

# 5. Identify the indices where anomaly errors are > optimal_threshold
# These are the anomaly images correctly classified as 'anomaly' (true positives)
true_positives_indices = np.where(anomaly_errors_full > optimal_threshold)[0]

# 6. Filter the images and errors based on these indices
detected_as_anomaly_originals_tp = [anomaly_originals[i] for i in true_positives_indices]
detected_as_anomaly_reconstructions_tp = [anomaly_reconstructions[i] for i in true_positives_indices]
detected_as_anomaly_errors_tp = [anomaly_errors_full[i] for i in true_positives_indices]
detected_as_anomaly_subtypes_tp = [anomaly_subtype_labels[i] for i in true_positives_indices]

logger.info(f"Found {len(detected_as_anomaly_originals_tp)} anomaly images correctly detected as 'anomaly' (True Positives, error > {optimal_threshold:.4f}).")
for subtype in sorted(set(detected_as_anomaly_subtypes_tp)):
    logger.info(f"  - {detected_as_anomaly_subtypes_tp.count(subtype)} from '{subtype}' (total {dict_type_num.get(subtype, 0)} images in test set)")

# 7. Visualize these filtered anomaly images (true positives)
visualize_filtered_reconstructions(
    detected_as_anomaly_originals_tp,
    detected_as_anomaly_reconstructions_tp,
    detected_as_anomaly_errors_tp,
    save_path=os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_true_positives.png'),
    num_images=5, # Display up to 5 such images
    title="Anomaly Images Correctly Detected as 'Anomaly' (True Positives)"
)

# 1. Get original, reconstructed images and errors for the entire good test dataset
good_originals, good_reconstructions, good_errors_full = get_reconstructions_and_errors_for_dataset(
    test_good_loader, model, device, loss_type=LOSS_FUNCTION_TYPE
)

# 2. Identify the indices where good errors are > optimal_threshold
# These are the good images that were incorrectly classified as 'anomaly' (false positives)
false_positives_indices = np.where(good_errors_full > optimal_threshold)[0]

# 3. Filter the images and errors based on these indices
detected_as_anomaly_originals = [good_originals[i] for i in false_positives_indices]
detected_as_anomaly_reconstructions = [good_reconstructions[i] for i in false_positives_indices]
detected_as_anomaly_errors = [good_errors_full[i] for i in false_positives_indices]

logger.info(f"Found {len(detected_as_anomaly_originals)} good images detected as 'anomaly' (error > {optimal_threshold:.4f}).")

# 4. Visualize these filtered good images (false positives)
visualize_filtered_reconstructions(
    detected_as_anomaly_originals,
    detected_as_anomaly_reconstructions,
    detected_as_anomaly_errors,
    save_path=os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_false_positives.png'),
    num_images=5, # Display up to 5 such images
    title="Good Images Detected as 'Anomaly' (False Positives)"
)

# 2. Identify the indices where good errors are <= optimal_threshold
# These are the good images correctly classified as 'good' (true negatives)
true_negatives_indices = np.where(good_errors_full <= optimal_threshold)[0]

# 3. Filter the images and errors based on these indices
detected_as_good_originals_tn = [good_originals[i] for i in true_negatives_indices]
detected_as_good_reconstructions_tn = [good_reconstructions[i] for i in true_negatives_indices]
detected_as_good_errors_tn = [good_errors_full[i] for i in true_negatives_indices]

logger.info(f"Found {len(detected_as_good_originals_tn)} good images correctly classified as 'good' (True Negatives, error <= {optimal_threshold:.4f}).")

# 4. Visualize these filtered good images (true negatives)
visualize_filtered_reconstructions(
    detected_as_good_originals_tn,
    detected_as_good_reconstructions_tn,
    detected_as_good_errors_tn,
    save_path=os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_true_negatives.png'),
    num_images=5, # Display up to 5 such images
    title="Good Images Correctly Classified as 'Good' (True Negatives)"
)
