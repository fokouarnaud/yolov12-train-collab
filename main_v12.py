"""
YOLOv12-Face - Script Principal d'Entraînement
Adaptation pour Google Colab avec architecture attention-centrique
"""

import os
import sys
import torch
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv12FaceTrainer:
    """Classe principale pour l'entraînement YOLOv12-Face"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or "/content/yolov12_face_config.yaml"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.config = self._load_config()
        
    def _load_config(self):
        """Charge la configuration d'entraînement"""
        default_config = {
            'model_size': 'n',  # n, s, m, l, x
            'epochs': 100,
            'batch_size': 16,
            'image_size': 640,
            'data_path': '/content/datasets/yolo_widerface/dataset.yaml',
            'project_path': '/content/runs/train',
            'name': 'yolov12_face',
            'patience': 20,
            'save_period': 10,
            'workers': 2,
            'cache': False,
            'amp': True,  # Automatic Mixed Precision
            'cos_lr': True,  # Cosine Learning Rate
            'close_mosaic': 10,
            
            # Augmentations spécifiques visages
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,  # Pas de rotation
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,  # Pas de flip vertical
                'fliplr': 0.5,  # Flip horizontal OK
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            },
            
            # Loss weights spécifiques visages
            'loss': {
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5
            },
            
            # YOLOv12 spécifique
            'yolov12': {
                'attention_type': 'area',
                'num_regions': 4,
                'flash_attention': True,
                'r_elan': True,
                'residual_scaling': True
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                default_config.update(custom_config)
        
        return default_config
    
    def setup_model(self):
        """Initialise le modèle YOLOv12-Face"""
        logger.info(f"🚀 Initialisation YOLOv12{self.config['model_size']} pour détection de visages")
        
        model_name = f"yolov12{self.config['model_size']}.yaml"
        
        try:
            self.model = YOLO(model_name)
            logger.info(f"✅ Modèle {model_name} chargé avec succès")
            logger.info(f"Device utilisé: {self.device}")
            
            # Afficher info modèle
            total_params = sum(p.numel() for p in self.model.model.parameters())
            logger.info(f"Paramètres totaux: {total_params:,}")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def prepare_training_config(self):
        """Prépare la configuration d'entraînement complète"""
        
        train_config = {
            'data': self.config['data_path'],
            'epochs': self.config['epochs'],
            'patience': self.config['patience'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['image_size'],
            'save': True,
            'save_period': self.config['save_period'],
            'cache': self.config['cache'],
            'device': self.device,
            'workers': self.config['workers'],
            'project': self.config['project_path'],
            'name': self.config['name'],
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',  # Meilleur pour attention
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': self.config['cos_lr'],
            'close_mosaic': self.config['close_mosaic'],
            'resume': False,
            'amp': self.config['amp'],
        }
        
        # Ajouter augmentations
        train_config.update(self.config['augmentation'])
        
        # Ajouter loss weights
        train_config.update(self.config['loss'])
        
        return train_config
    
    def train(self):
        """Lance l'entraînement du modèle"""
        if self.model is None:
            self.setup_model()
        
        logger.info("🎯 Démarrage de l'entraînement YOLOv12-Face")
        
        # Configuration d'entraînement
        train_config = self.prepare_training_config()
        
        # Afficher configuration
        logger.info("Configuration d'entraînement:")
        for key, value in train_config.items():
            if isinstance(value, (int, float, str, bool)):
                logger.info(f"  {key}: {value}")
        
        try:
            # Lancer entraînement
            results = self.model.train(**train_config)
            
            logger.info("✅ Entraînement terminé avec succès")
            
            # Sauvegarder configuration utilisée
            config_save_path = f"{train_config['project']}/{train_config['name']}/config_used.yaml"
            os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
            with open(config_save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur pendant l'entraînement: {e}")
            raise
    
    def evaluate(self, weights_path=None):
        """Évalue le modèle entraîné"""
        if weights_path is None:
            weights_path = f"{self.config['project_path']}/{self.config['name']}/weights/best.pt"
        
        if not os.path.exists(weights_path):
            logger.error(f"❌ Poids non trouvés: {weights_path}")
            return None
        
        logger.info(f"📊 Évaluation du modèle: {weights_path}")
        
        # Charger modèle avec poids entraînés
        model = YOLO(weights_path)
        
        # Configuration évaluation
        eval_config = {
            'data': self.config['data_path'],
            'imgsz': self.config['image_size'],
            'batch': 8,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': True,
            'device': self.device,
            'dnn': False,
            'plots': True,
            'save_json': True,
            'save_hybrid': False,
            'verbose': True
        }
        
        try:
            # Évaluation
            metrics = model.val(**eval_config)
            
            # Afficher résultats
            logger.info("🏆 Résultats d'évaluation:")
            logger.info(f"  mAP50: {metrics.box.map50:.3f}")
            logger.info(f"  mAP50-95: {metrics.box.map:.3f}")
            logger.info(f"  Precision: {metrics.box.mp:.3f}")
            logger.info(f"  Recall: {metrics.box.mr:.3f}")
            
            # Analyse par taille si disponible
            if hasattr(metrics.box, 'maps'):
                logger.info("  Détail par classe:")
                for i, map_val in enumerate(metrics.box.maps):
                    logger.info(f"    Classe {i}: {map_val:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur pendant évaluation: {e}")
            return None
    
    def export_model(self, weights_path=None, formats=['onnx']):
        """Exporte le modèle dans différents formats"""
        if weights_path is None:
            weights_path = f"{self.config['project_path']}/{self.config['name']}/weights/best.pt"
        
        if not os.path.exists(weights_path):
            logger.error(f"❌ Poids non trouvés: {weights_path}")
            return
        
        logger.info(f"📦 Export du modèle: {weights_path}")
        
        # Charger modèle
        model = YOLO(weights_path)
        
        for format_type in formats:
            try:
                logger.info(f"  Export format: {format_type}")
                
                export_config = {
                    'format': format_type,
                    'imgsz': self.config['image_size'],
                    'half': True,
                    'dynamic': False,
                    'simplify': True
                }
                
                if format_type == 'onnx':
                    export_config['opset'] = 17
                
                model.export(**export_config)
                logger.info(f"  ✅ Export {format_type} terminé")
                
            except Exception as e:
                logger.error(f"  ❌ Erreur export {format_type}: {e}")
    
    def run_inference_test(self, image_path=None, weights_path=None):
        """Test d'inférence sur une image"""
        if weights_path is None:
            weights_path = f"{self.config['project_path']}/{self.config['name']}/weights/best.pt"
        
        if not os.path.exists(weights_path):
            logger.error(f"❌ Poids non trouvés: {weights_path}")
            return
        
        # Image de test par défaut
        if image_path is None:
            image_path = "/content/datasets/yolo_widerface/images/val"
            # Prendre la première image trouvée
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(Path(image_path).glob(ext))
                if images:
                    image_path = str(images[0])
                    break
        
        if not os.path.exists(image_path):
            logger.error(f"❌ Image non trouvée: {image_path}")
            return
        
        logger.info(f"🖼️ Test d'inférence: {image_path}")
        
        # Charger modèle
        model = YOLO(weights_path)
        
        try:
            # Inférence
            results = model(image_path, conf=0.25, iou=0.45)
            
            # Afficher résultats
            for r in results:
                logger.info(f"  Détections: {len(r.boxes)}")
                
                for i, box in enumerate(r.boxes):
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w, h = x2-x1, y2-y1
                    logger.info(f"    Visage {i+1}: conf={conf:.3f}, size={w:.0f}x{h:.0f}")
                
                # Sauvegarder image avec détections
                output_path = "/content/inference_result.jpg"
                im_array = r.plot()
                import cv2
                cv2.imwrite(output_path, im_array)
                logger.info(f"  Résultat sauvé: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur inférence: {e}")
            return None


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12-Face Training')
    parser.add_argument('--config', type=str, help='Chemin fichier configuration')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-only', action='store_true', help='Évaluation uniquement')
    parser.add_argument('--export-only', action='store_true', help='Export uniquement')
    parser.add_argument('--test-inference', action='store_true', help='Test inférence uniquement')
    parser.add_argument('--weights', type=str, help='Chemin poids pour éval/export/test')
    
    args = parser.parse_args()
    
    # Créer trainer
    trainer = YOLOv12FaceTrainer(args.config)
    
    # Mettre à jour config avec arguments
    if args.model_size:
        trainer.config['model_size'] = args.model_size
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['batch_size'] = args.batch_size
    
    # Exécuter selon mode
    if args.eval_only:
        trainer.evaluate(args.weights)
    elif args.export_only:
        trainer.export_model(args.weights)
    elif args.test_inference:
        trainer.run_inference_test(weights_path=args.weights)
    else:
        # Entraînement complet
        trainer.train()
        
        # Évaluation automatique après entraînement
        trainer.evaluate()
        
        # Export automatique
        trainer.export_model()
        
        # Test inférence
        trainer.run_inference_test()


if __name__ == "__main__":
    main()
