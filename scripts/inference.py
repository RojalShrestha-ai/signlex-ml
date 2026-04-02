#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import ASL_LABELS, ASL_LABELS_INV
from data.dataset import get_val_transforms
from models.vit_pretrained import create_pretrained_vit
from utils.visualization import visualize_attention, get_attention_maps


class ASLPredictor:

    
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "large",
        device: str = "cuda",
    ):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create model
        print(f"Loading ViT-{model_size.upper()} model...")
        self.model = create_pretrained_vit(
            model_size=model_size,
            num_classes=26,
            pretrained=False,  # We'll load our own weights
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = get_val_transforms(224)
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict ASL sign from image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of (letter, probability) tuples
        """
        # Preprocess
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(tensor)
        probs = F.softmax(output, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k)
        
        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            letter = ASL_LABELS_INV[idx]
            results.append((letter, prob))
        
        return results
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Image.Image],
    ) -> List[List[Tuple[str, float]]]:
 
        tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensors.append(self.transform(img))
        
        batch = torch.stack(tensors).to(self.device)
   
        outputs = self.model(batch)
        probs = F.softmax(outputs, dim=1)
        

        results = []
        for i in range(len(images)):
            top_probs, top_indices = probs[i].topk(5)
            preds = [(ASL_LABELS_INV[idx.item()], prob.item()) 
                     for prob, idx in zip(top_probs, top_indices)]
            results.append(preds)
        
        return results
    
    @torch.no_grad()
    def predict_with_attention(
        self,
        image: Image.Image,
    ) -> Tuple[List[Tuple[str, float]], torch.Tensor]:

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get attention maps
        attention = get_attention_maps(self.model, tensor, layer_idx=-1)
        
        # Forward pass
        output = self.model(tensor)
        probs = F.softmax(output, dim=1)[0]
        
        # Get top-5 predictions
        top_probs, top_indices = probs.topk(5)
        results = [(ASL_LABELS_INV[idx.item()], prob.item()) 
                   for prob, idx in zip(top_probs, top_indices)]
        
        return results, attention


def run_webcam_inference(predictor: ASLPredictor):
    """
    Run real-time inference on webcam feed.
    """
    import cv2
    
    print("\nStarting webcam...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Predict every 3rd frame for performance
        if frame_count % 3 == 0:
            predictions = predictor.predict(pil_image, top_k=3)
            
            # Display predictions
            y_offset = 30
            for letter, prob in predictions:
                text = f"{letter}: {prob:.2%}"
                cv2.putText(
                    frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2
                )
                y_offset += 30
        
        # Show frame
        cv2.imshow('SignLex ASL Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'capture_{frame_count}.jpg', frame)
            print(f"Saved capture_{frame_count}.jpg")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def run_batch_inference(predictor: ASLPredictor, folder_path: str):
    """
    Run inference on a folder of images.
    """
    from tqdm import tqdm
    
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    image_paths = [
        p for p in folder.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"\nFound {len(image_paths)} images in {folder}")
    
    results = {}
    
    for img_path in tqdm(image_paths, desc="Processing"):
        image = Image.open(img_path)
        predictions = predictor.predict(image, top_k=3)
        
        results[str(img_path)] = {
            'predictions': [
                {'letter': letter, 'confidence': prob}
                for letter, prob in predictions
            ]
        }
    
    # Save results
    output_path = folder / 'predictions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # Print summary
    print("\nSample predictions:")
    for path, data in list(results.items())[:5]:
        top_pred = data['predictions'][0]
        print(f"  {Path(path).name}: {top_pred['letter']} ({top_pred['confidence']:.2%})")


def main():
    parser = argparse.ArgumentParser(description="SignLex Inference")
    
    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single image')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam')
    input_group.add_argument('--batch', type=str, help='Path to folder of images')
    
    # Model settings
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model_size', type=str, default='large',
                        choices=['base', 'large', 'huge'])
    parser.add_argument('--device', type=str, default='cuda')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Show attention visualization')
    parser.add_argument('--save', type=str, help='Save visualization to path')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ASLPredictor(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        device=args.device,
    )
    
    # Run inference
    if args.webcam:
        run_webcam_inference(predictor)
    
    elif args.batch:
        run_batch_inference(predictor, args.batch)
    
    elif args.image:
        print(f"\nProcessing: {args.image}")
        image = Image.open(args.image)
        
        if args.visualize:
            predictions, attention = predictor.predict_with_attention(image)
            
            # Show attention
            tensor = predictor.transform(image.convert('RGB'))
            fig = visualize_attention(
                tensor, attention,
                save_path=args.save or 'attention_visualization.png'
            )
        else:
            predictions = predictor.predict(image, top_k=5)
        
        # Print results
        print("\nPredictions:")
        for rank, (letter, prob) in enumerate(predictions, 1):
            print(f"  {rank}. {letter}: {prob:.2%}")


if __name__ == "__main__":
    main()
