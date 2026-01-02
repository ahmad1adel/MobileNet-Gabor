from src_mobilenet_gabor_both import FaceRecognitionPipeline
import os
import numpy as np

def train_pipeline():
    pipeline = FaceRecognitionPipeline(
        remove_bg=False,
        similarity_threshold=0.55
    )
    
    # Define dataset paths
    datasets = [
        {'train': 'dataset/masked/train', 'val': 'dataset/masked/val'},
        {'train': 'dataset/unmasked/train', 'val': 'dataset/unmasked/val'}
    ]
    
    all_train_embeddings = []
    all_train_gabor = []
    all_train_labels = []
    
    all_val_embeddings = []
    all_val_gabor = []
    all_val_labels = []

    print("Preparing training data from multiple sources...")
    for ds in datasets:
        if os.path.exists(ds['train']):
            print(f"Processing {ds['train']}...")
            e, g, l = pipeline.prepare_training_data(ds['train'])
            all_train_embeddings.append(e)
            all_train_gabor.append(g)
            all_train_labels.extend(l)
            
        if os.path.exists(ds['val']):
            print(f"Processing {ds['val']}...")
            e, g, l = pipeline.prepare_training_data(ds['val'])
            all_val_embeddings.append(e)
            all_val_gabor.append(g)
            all_val_labels.extend(l)

    if not all_train_labels:
        print("No training data found.")
        return

    # Combine data
    train_embeddings = np.vstack(all_train_embeddings)
    train_gabor = np.vstack(all_train_gabor)
    
    print(f"Training on combined dataset: {len(all_train_labels)} samples...")
    pipeline.detector.train(train_embeddings, train_gabor, all_train_labels)
    
    if all_val_labels:
        val_embeddings = np.vstack(all_val_embeddings)
        val_gabor = np.vstack(all_val_gabor)
        metrics = pipeline.detector.evaluate(val_embeddings, val_gabor, all_val_labels)
        print(f"Combined Validation Accuracy: {metrics['accuracy']:.4f}")
    
    pipeline.save_pipeline('models/mobilenet_gabor_both')
    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
