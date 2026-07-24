import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns

# Import the PaDiM model
from model import PaDiM
from dataset import MVTecDataset

# Define image dimensions and hyperparameters (must match training)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
D_REDUCED = 100 # must match train.py
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Data specific configuration
target_object = 'wood' # Must match training object
detect_category = '' # default: empty string, '' for all anomalies, or specify an anomaly type

# Define directory for saving models
MODEL_SAVE_DIR = './saved_models/' + target_object

# Define postfix for model filename. No epoch suffix -- PaDiM has no epochs,
# see train.py.
model_filename = f'padim_{target_object}'
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
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


def denormalize(img_chw):
    # Reverses the ImageNet Normalize above so images can be displayed;
    # img_chw is a (3, H, W) numpy array.
    img = img_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)

# Helper function to calculate anomaly scores
def calculate_anomaly_scores(dataloader, model, device, image_size=IMAGE_SIZE):
    scores = []
    if dataloader is None:
        return np.array([])

    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            image_scores, _ = model.predict(inputs, device, image_size)
            scores.extend(image_scores.cpu().numpy())
    return np.array(scores)

# Helper function to get original images, anomaly maps, and scores from a dataloader
def get_scores_and_maps_for_dataset(dataloader, model, device, image_size=IMAGE_SIZE):
    original_images_list = []
    anomaly_maps_list = []
    scores_list = []

    if dataloader is None:
        return [], [], np.array([])

    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            image_scores, anomaly_maps = model.predict(inputs, device, image_size)

            original_images_list.extend([denormalize(img.cpu().numpy()) for img in inputs])
            anomaly_maps_list.extend([m.cpu().numpy() for m in anomaly_maps])
            scores_list.extend(image_scores.cpu().numpy())

    return original_images_list, anomaly_maps_list, np.array(scores_list)

# Helper function to visualize a list of original images with their anomaly heatmaps
def visualize_filtered_anomaly_maps(original_imgs, anomaly_maps, scores, save_path, num_images=5, title=""):
    if not original_imgs:
        logger.info(f"No images to visualize for {title}.")
        return

    num_to_display = min(num_images, len(original_imgs))
    plt.figure(figsize=(num_to_display * 2.5, 7))
    for i in range(num_to_display):
        # Original Image
        plt.subplot(3, num_to_display, i + 1)
        plt.imshow(original_imgs[i])
        plt.title("Original")
        plt.axis('off')

        # Anomaly Heatmap
        plt.subplot(3, num_to_display, num_to_display + i + 1)
        plt.imshow(anomaly_maps[i], cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Anomaly Map ({scores[i]:.4f})")
        plt.axis('off')

        # Overlay
        plt.subplot(3, num_to_display, 2 * num_to_display + i + 1)
        plt.imshow(original_imgs[i])
        plt.imshow(anomaly_maps[i], cmap='hot', alpha=0.5)
        plt.title("Overlay")
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

# Load the fitted model
model_load_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename}.pth')

if os.path.exists(model_load_path):
    model = PaDiM.load(model_load_path, device=device)
    logger.info(f"Loaded model from {model_load_path}")
else:
    logger.error(f"Model not found at {model_load_path}. Please ensure training was successful.")
    exit()

# Calculate anomaly scores for good test images
good_scores = calculate_anomaly_scores(test_good_loader, model, device)

# Calculate scores per anomaly type, then pool them for the aggregate statistics below
anomaly_scores_by_type = {
    subtype: calculate_anomaly_scores(loader, model, device)
    for subtype, loader in anomaly_loaders_by_type.items()
}
anomaly_scores = np.concatenate(list(anomaly_scores_by_type.values())) if anomaly_scores_by_type else np.array([])

logger.info(f"Mean anomaly score for good images: {np.mean(good_scores):.4f}")
for subtype, scores in anomaly_scores_by_type.items():
    logger.info(f"Mean anomaly score for '{subtype}' anomaly images: {np.mean(scores):.4f} (n={len(scores)})")
logger.info(f"Mean anomaly score for anomaly images (all types combined): {np.mean(anomaly_scores):.4f}")

# Visualize Anomaly Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(good_scores, bins=50, alpha=0.7, label='Good Images (Anomaly Score)', color='blue')
plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly Images (Anomaly Score)', color='red')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score (max Mahalanobis distance)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
score_distribution_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_score_distribution.png')
plt.savefig(score_distribution_path)
plt.close()
logger.info(f"Saved plot to {score_distribution_path}")

# Determine Optimal Anomaly Threshold (ROC Curve)
y_true = np.concatenate((np.zeros(len(good_scores)), np.ones(len(anomaly_scores))))
y_scores = np.concatenate((good_scores, anomaly_scores))

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


# 1. Get original images, anomaly maps and scores per anomaly type, then pool them
# (keeping a parallel subtype label per image so false negatives can be broken down by type)
anomaly_originals = []
anomaly_maps_all = []
anomaly_scores_full_list = []
anomaly_subtype_labels = []
for subtype, loader in anomaly_loaders_by_type.items():
    originals, maps, scores_full = get_scores_and_maps_for_dataset(loader, model, device)
    anomaly_originals.extend(originals)
    anomaly_maps_all.extend(maps)
    anomaly_scores_full_list.extend(scores_full)
    anomaly_subtype_labels.extend([subtype] * len(scores_full))
anomaly_scores_full = np.array(anomaly_scores_full_list)

# 2. Identify the indices where anomaly scores are <= optimal_threshold
# These are the anomaly images that were incorrectly classified as 'good' (false negatives)
false_negatives_indices = np.where(anomaly_scores_full <= optimal_threshold)[0]

# 3. Filter the images and scores based on these indices
detected_as_good_originals = [anomaly_originals[i] for i in false_negatives_indices]
detected_as_good_maps = [anomaly_maps_all[i] for i in false_negatives_indices]
detected_as_good_scores = [anomaly_scores_full[i] for i in false_negatives_indices]
detected_as_good_subtypes = [anomaly_subtype_labels[i] for i in false_negatives_indices]

logger.info(f"Found {len(detected_as_good_originals)} anomaly images detected as 'good' (score <= {optimal_threshold:.4f}).")
for subtype in sorted(set(detected_as_good_subtypes)):
    logger.info(f"  - {detected_as_good_subtypes.count(subtype)} from '{subtype}' (total {dict_type_num.get(subtype, 0)} images in test set)")

# 4. Visualize these filtered anomaly images (false negatives)
visualize_filtered_anomaly_maps(
    detected_as_good_originals,
    detected_as_good_maps,
    detected_as_good_scores,
    save_path=os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_false_negatives.png'),
    num_images=5, # Display up to 5 such images
    title="Anomaly Images Detected as 'Good' (False Negatives)"
)

# 1. Get original images, anomaly maps and scores for the entire good test dataset
good_originals, good_maps, good_scores_full = get_scores_and_maps_for_dataset(test_good_loader, model, device)

# 2. Identify the indices where good scores are > optimal_threshold
# These are the good images that were incorrectly classified as 'anomaly' (false positives)
false_positives_indices = np.where(good_scores_full > optimal_threshold)[0]

# 3. Filter the images and scores based on these indices
detected_as_anomaly_originals = [good_originals[i] for i in false_positives_indices]
detected_as_anomaly_maps = [good_maps[i] for i in false_positives_indices]
detected_as_anomaly_scores = [good_scores_full[i] for i in false_positives_indices]

logger.info(f"Found {len(detected_as_anomaly_originals)} good images detected as 'anomaly' (score > {optimal_threshold:.4f}).")

# 4. Visualize these filtered good images (false positives)
visualize_filtered_anomaly_maps(
    detected_as_anomaly_originals,
    detected_as_anomaly_maps,
    detected_as_anomaly_scores,
    save_path=os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_false_positives.png'),
    num_images=5, # Display up to 5 such images
    title="Good Images Detected as 'Anomaly' (False Positives)"
)
