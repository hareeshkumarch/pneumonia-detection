PneumoNet: Privacy-Preserving Pneumonia Detection with Federated Learning

Overview

PneumoNet is an innovative system designed to revolutionize pneumonia detection using federated learning and Convolutional Neural Networks (CNNs), achieving a remarkable 92.5% test accuracy as of June 21, 2025. This privacy-preserving solution enables decentralized training across multiple healthcare institutions, ensuring compliance with regulations like HIPAA and GDPR by avoiding central data storage. Utilizing a dataset of 5,000 chest X-ray images (2,500 normal, 2,500 pneumonia) from five institutions, it employs horizontal and vertical federated learning to leverage diverse data types. The system preprocesses images to 128x128 pixels, applies normalization, and enhances models with local augmentation. Powered by PySyft and FATE, PneumoNet offers a secure user interface for real-time image analysis and report generation. Trained on NVIDIA Tesla V100 GPUs, it addresses diagnostic challenges in high-risk populations and aims to scale globally. Future enhancements include optimization for low-spec devices and integration of multimodal data like electronic health records for personalized care.

Features





High accuracy: 95.2% training, 93.8% validation.



Privacy-compliant with HIPAA and GDPR via federated learning.



User interface for secure image upload and report generation.



Powered by PySyft and FATE for federated learning collaboration.



Supports horizontal and vertical federated learning for diverse data.

Usage





Distribute global model to nodes using federated learning.



Train local models on X-ray data.



Encrypt and aggregate updates via federated learning averaging.



Analyze real-time predictions with downloadable medical reports.

Additional Info





Dataset: 5,000 chest X-ray images (2,500 normal, 2,500 pneumonia) from 5 institutions.



Preprocessing: Images resized to 128x128 pixels, normalized, with local augmentation.



Hardware: Trained on NVIDIA Tesla V100 GPUs.



Future Work: Enhance for low-spec devices and integrate multimodal data like EHRs.
