# EfficientNetB1-Multi-Modal-HAM10000
This project introduces a comprehensive multi-modal deep learning framework for automated skin lesion classification using the HAM10000 (Human Against Machine with 10000 training images) dataset, a widely used benchmark in dermatology and medical image analysis. The study aims to address two major challenges in skin cancer detection: high inter-class imbalance and the limited representational capacity of image-only models when clinical context is ignored. The workflow begins with careful dataset preparation, where dermoscopic images are systematically linked to their corresponding clinical metadata. Missing age values are imputed using the dataset mean, categorical variables such as patient sex and lesion localization are transformed via one-hot encoding, and numerical features are standardized to ensure stable and efficient optimization. The dataset is then divided into stratified training and validation subsets (80/20 split) to preserve the original diagnostic distribution across seven lesion categories. To further counteract class imbalance, class weights are computed and incorporated directly into the training loss.

For the modeling stage, a pretrained EfficientNetB1 architecture, initialized with ImageNet weights, is adopted as a high-capacity yet computationally efficient convolutional backbone for dermoscopic image feature extraction. The convolutional layers are kept frozen to retain robust low- and mid-level visual representations while reducing overfitting and training cost. Image features are condensed using global average pooling and regularized via dropout. In parallel, a dedicated metadata processing branch based on a lightweight multilayer perceptron learns discriminative patterns from structured clinical attributes. These two complementary feature streams are then integrated through feature-level fusion, allowing the network to jointly reason over visual appearance and patient-specific clinical information before final classification.

The network is trained end-to-end using the Adam optimizer, sparse categorical cross-entropy loss, and an early stopping strategy to prevent overfitting. Performance is evaluated on a held-out validation set using accuracy and class-sensitive metrics. The final model achieves a validation accuracy of 74.64%, a weighted F1-score of 0.76, and a macro F1-score of 0.59, reflecting a strong balance between overall correctness and minority-class sensitivity. Notably, the model demonstrates improved recall for clinically critical but underrepresented classes such as melanoma (mel), basal cell carcinoma (bcc), and vascular lesions (vasc), while maintaining high precision for the dominant nevus (nv) class. These results highlight the effectiveness of multi-modal learning in dermatological diagnosis and confirm that incorporating structured clinical metadata alongside dermoscopic images leads to more robust, clinically meaningful, and generalizable skin lesion classification performance.
Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Loaded the HAM10000_metadata.csv file and created absolute image paths by mapping image_id → jpg.

Cleaned and prepared metadata:

Filled missing age values with the mean

One-hot encoded categorical features (sex, localization)

Standardized age using StandardScaler

Created numeric labels using class_names = sorted(dx.unique()) and a label_map.

Split the dataset into 80/20 train/validation using stratified split.

Computed class weights to mitigate severe class imbalance (e.g., nv dominates the dataset).

Built a multi-input tf.data pipeline that yields:

{"image_input": image_tensor, "meta_input": meta_tensor}, label

Built a late-fusion multi-modal architecture:

Image branch: EfficientNetB1 (ImageNet pretrained, include_top=False, frozen) → GAP → Dropout(0.3)

Metadata branch: Dense(32) → Dense(16)

Concatenate(image_features, meta_features) → Dropout(0.4) → Dense(64) → Softmax(7)

Trained using:

Adam (lr=1e-3)

sparse categorical cross-entropy

early stopping (patience=5, restore best weights)

class weights
