"""
YOLOv12-Face - Configuration et Hyperparamètres
Configuration optimisée pour l'architecture attention-centrique
"""

import yaml
import os
import logging
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv12FaceConfig:
    """Gestionnaire de configuration pour YOLOv12-Face"""
    
    def __init__(self):
        self.base_config = self._get_base_config()
        self.model_configs = self._get_model_configs()
        self.training_configs = self._get_training_configs()
        self.augmentation_configs = self._get_augmentation_configs()
    
    def _get_base_config(self):
        """Configuration de base YOLOv12-Face"""
        return {
            'project_name': 'yolov12_face',
            'description': 'YOLOv12-Face avec architecture attention-centrique',
            'version': '1.0',
            'framework': 'ultralytics',
            
            # Chemins par défaut
            'paths': {
                'data': '/content/datasets/yolo_widerface/dataset.yaml',
                'project': '/content/runs/train',
                'export': '/content/runs/export',
                'weights': '/content/runs/train/yolov12_face/weights'
            },
            
            # Configuration YOLOv12 spécifique
            'architecture': {
                'type': 'yolov12',
                'attention_mechanism': 'area',
                'backbone': 'attention_centric',
                'neck': 'r_elan',
                'head': 'multi_scale'
            }
        }
    
    def _get_model_configs(self):
        """Configurations pour différentes tailles de modèles"""
        return {
            'n': {  # Nano - Optimisé pour Colab
                'depth_multiple': 0.33,
                'width_multiple': 0.25,
                'max_channels': 1024,
                'scales': {
                    'backbone': [0.33, 0.25],
                    'neck': [0.33, 0.25],
                    'head': [0.33, 0.25]
                },
                'attention_config': {
                    'num_regions': 4,
                    'flash_attention': False,  # Trop lourd pour nano
                    'attention_dropout': 0.1
                },
                'recommended_batch': 16,
                'memory_usage': 'low'
            },
            
            's': {  # Small
                'depth_multiple': 0.33,
                'width_multiple': 0.50,
                'max_channels': 1024,
                'scales': {
                    'backbone': [0.33, 0.50],
                    'neck': [0.33, 0.50],
                    'head': [0.33, 0.50]
                },
                'attention_config': {
                    'num_regions': 4,
                    'flash_attention': True,
                    'attention_dropout': 0.1
                },
                'recommended_batch': 12,
                'memory_usage': 'medium'
            },
            
            'm': {  # Medium
                'depth_multiple': 0.67,
                'width_multiple': 0.75,
                'max_channels': 768,
                'scales': {
                    'backbone': [0.67, 0.75],
                    'neck': [0.67, 0.75],
                    'head': [0.67, 0.75]
                },
                'attention_config': {
                    'num_regions': 6,
                    'flash_attention': True,
                    'attention_dropout': 0.1
                },
                'recommended_batch': 8,
                'memory_usage': 'high'
            },
            
            'l': {  # Large
                'depth_multiple': 1.0,
                'width_multiple': 1.0,
                'max_channels': 512,
                'scales': {
                    'backbone': [1.0, 1.0],
                    'neck': [1.0, 1.0],
                    'head': [1.0, 1.0]
                },
                'attention_config': {
                    'num_regions': 8,
                    'flash_attention': True,
                    'attention_dropout': 0.05
                },
                'recommended_batch': 4,
                'memory_usage': 'very_high'
            },
            
            'x': {  # Extra Large
                'depth_multiple': 1.33,
                'width_multiple': 1.25,
                'max_channels': 512,
                'scales': {
                    'backbone': [1.33, 1.25],
                    'neck': [1.33, 1.25],
                    'head': [1.33, 1.25]
                },
                'attention_config': {
                    'num_regions': 8,
                    'flash_attention': True,
                    'attention_dropout': 0.05
                },
                'recommended_batch': 2,
                'memory_usage': 'extreme'
            }
        }
    
    def _get_training_configs(self):
        """Configurations d'entraînement optimisées"""
        return {
            'quick_test': {  # Test rapide
                'epochs': 10,
                'patience': 5,
                'save_period': 5,
                'val_period': 2,
                'description': 'Configuration pour tests rapides'
            },
            
            'development': {  # Développement
                'epochs': 50,
                'patience': 10,
                'save_period': 10,
                'val_period': 5,
                'description': 'Configuration pour développement'
            },
            
            'production': {  # Production
                'epochs': 300,
                'patience': 50,
                'save_period': 25,
                'val_period': 10,
                'description': 'Configuration pour entraînement complet'
            },
            
            'fine_tuning': {  # Fine-tuning
                'epochs': 100,
                'patience': 20,
                'save_period': 10,
                'val_period': 5,
                'lr_factor': 0.1,  # LR réduit pour fine-tuning
                'description': 'Configuration pour fine-tuning'
            }
        }
    
    def _get_augmentation_configs(self):
        """Configurations d'augmentation spécialisées visages"""
        return {
            'conservative': {  # Augmentation conservatrice
                'hsv_h': 0.010,
                'hsv_s': 0.5,
                'hsv_v': 0.3,
                'degrees': 0.0,
                'translate': 0.05,
                'scale': 0.3,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.3,
                'mosaic': 0.5,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'description': 'Augmentation conservatrice pour visages'
            },
            
            'moderate': {  # Augmentation modérée (recommandée)
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'description': 'Augmentation modérée optimale pour visages'
            },
            
            'aggressive': {  # Augmentation agressive
                'hsv_h': 0.020,
                'hsv_s': 0.9,
                'hsv_v': 0.5,
                'degrees': 0.0,  # Toujours 0 pour visages
                'translate': 0.15,
                'scale': 0.7,
                'shear': 0.0,  # Toujours 0 pour visages
                'perspective': 0.0,  # Toujours 0 pour visages
                'flipud': 0.0,  # Toujours 0 pour visages
                'fliplr': 0.7,
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'description': 'Augmentation agressive pour datasets limités'
            }
        }
    
    def get_complete_config(self, model_size='n', training_mode='development', 
                          augmentation_mode='moderate', custom_overrides=None):
        """Génère une configuration complète"""
        
        if model_size not in self.model_configs:
            raise ValueError(f"Taille modèle non supportée: {model_size}")
        
        if training_mode not in self.training_configs:
            raise ValueError(f"Mode entraînement non supporté: {training_mode}")
        
        if augmentation_mode not in self.augmentation_configs:
            raise ValueError(f"Mode augmentation non supporté: {augmentation_mode}")
        
        # Configuration de base
        config = self.base_config.copy()
        
        # Ajouter configuration modèle
        config['model'] = self.model_configs[model_size].copy()
        config['model']['size'] = model_size
        
        # Ajouter configuration entraînement
        config['training'] = self.training_configs[training_mode].copy()
        config['training']['mode'] = training_mode
        
        # Ajouter configuration augmentation
        config['augmentation'] = self.augmentation_configs[augmentation_mode].copy()
        config['augmentation']['mode'] = augmentation_mode
        
        # Paramètres optimisés pour YOLOv12-Face
        config['hyperparameters'] = self._get_optimized_hyperparameters(model_size)
        
        # Appliquer overrides personnalisés
        if custom_overrides:
            config = self._apply_overrides(config, custom_overrides)
        
        return config
    
    def _get_optimized_hyperparameters(self, model_size):
        """Hyperparamètres optimisés selon la taille du modèle"""
        
        base_params = {
            # Optimisation
            'optimizer': 'AdamW',  # Meilleur pour attention
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights spécifiques visages
            'box': 7.5,  # Augmenté pour visages
            'cls': 0.5,
            'dfl': 1.5,
            
            # Paramètres YOLOv12
            'cos_lr': True,  # Cosine learning rate
            'close_mosaic': 10,  # Fermer mosaic avant fin
            'amp': True,  # Mixed precision
            'single_cls': False,
            'rect': False,
            'deterministic': True,
            'seed': 42,
            
            # Spécifique attention
            'attention_lr_scale': 0.5,  # LR réduit pour attention
            'attention_weight_decay': 0.0001,
        }
        
        # Ajustements selon taille modèle
        size_adjustments = {
            'n': {
                'lr0': 0.01,
                'batch_size': 16,
                'accumulate': 1,
                'workers': 2
            },
            's': {
                'lr0': 0.008,
                'batch_size': 12,
                'accumulate': 2,
                'workers': 4
            },
            'm': {
                'lr0': 0.006,
                'batch_size': 8,
                'accumulate': 4,
                'workers': 4
            },
            'l': {
                'lr0': 0.004,
                'batch_size': 4,
                'accumulate': 8,
                'workers': 6
            },
            'x': {
                'lr0': 0.002,
                'batch_size': 2,
                'accumulate': 16,
                'workers': 8
            }
        }
        
        if model_size in size_adjustments:
            base_params.update(size_adjustments[model_size])
        
        return base_params
    
    def _apply_overrides(self, config, overrides):
        """Applique des overrides personnalisés"""
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config, overrides)
        return config
    
    def save_config(self, config, output_path):
        """Sauvegarde la configuration en YAML"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Configuration sauvegardée: {output_path}")
    
    def load_config(self, config_path):
        """Charge une configuration depuis un fichier YAML"""
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration non trouvée: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Configuration chargée: {config_path}")
        return config
    
    def create_training_command(self, config):
        """Génère la commande d'entraînement"""
        
        model_size = config['model']['size']
        hyperparams = config['hyperparameters']
        augmentation = config['augmentation']
        training = config['training']
        
        # Base command
        cmd_parts = [
            f"python main_v12.py",
            f"--model-size {model_size}",
            f"--epochs {training['epochs']}",
            f"--batch-size {hyperparams['batch_size']}",
        ]
        
        # Ajouter paramètres optionnels
        if 'patience' in training:
            cmd_parts.append(f"--patience {training['patience']}")
        
        command = " ".join(cmd_parts)
        
        logger.info(f"Commande d'entraînement: {command}")
        return command
    
    def validate_config(self, config):
        """Valide une configuration"""
        
        required_keys = ['model', 'training', 'augmentation', 'hyperparameters']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Clé manquante dans config: {key}")
        
        # Vérifier taille modèle
        if config['model']['size'] not in self.model_configs:
            raise ValueError(f"Taille modèle invalide: {config['model']['size']}")
        
        # Vérifier hyperparamètres
        hyp = config['hyperparameters']
        if hyp['lr0'] <= 0 or hyp['lr0'] > 1:
            raise ValueError(f"Learning rate invalide: {hyp['lr0']}")
        
        if hyp['batch_size'] <= 0:
            raise ValueError(f"Batch size invalide: {hyp['batch_size']}")
        
        logger.info("✅ Configuration validée")
        return True


def create_preset_configs():
    """Crée des configurations prédéfinies"""
    
    config_manager = YOLOv12FaceConfig()
    presets = {}
    
    # Configuration Colab optimisée
    presets['colab_nano'] = config_manager.get_complete_config(
        model_size='n',
        training_mode='development',
        augmentation_mode='moderate',
        custom_overrides={
            'paths': {
                'data': '/content/datasets/yolo_widerface/dataset.yaml',
                'project': '/content/runs/train'
            },
            'hyperparameters': {
                'batch_size': 16,
                'workers': 2,
                'cache': False
            }
        }
    )
    
    # Configuration production
    presets['production_small'] = config_manager.get_complete_config(
        model_size='s',
        training_mode='production',
        augmentation_mode='moderate',
        custom_overrides={
            'training': {
                'epochs': 300,
                'patience': 50
            }
        }
    )
    
    # Configuration fine-tuning
    presets['fine_tune'] = config_manager.get_complete_config(
        model_size='s',
        training_mode='fine_tuning',
        augmentation_mode='conservative',
        custom_overrides={
            'hyperparameters': {
                'lr0': 0.001,  # LR réduit
                'warmup_epochs': 1
            }
        }
    )
    
    return presets


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration YOLOv12-Face')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--training-mode', type=str, default='development',
                       choices=['quick_test', 'development', 'production', 'fine_tuning'])
    parser.add_argument('--augmentation-mode', type=str, default='moderate',
                       choices=['conservative', 'moderate', 'aggressive'])
    parser.add_argument('--output', type=str, default='/content/yolov12_face_config.yaml')
    parser.add_argument('--preset', type=str, choices=['colab_nano', 'production_small', 'fine_tune'])
    
    args = parser.parse_args()
    
    config_manager = YOLOv12FaceConfig()
    
    if args.preset:
        # Utiliser preset
        presets = create_preset_configs()
        config = presets[args.preset]
        logger.info(f"🎯 Utilisation preset: {args.preset}")
    else:
        # Créer configuration personnalisée
        config = config_manager.get_complete_config(
            model_size=args.model_size,
            training_mode=args.training_mode,
            augmentation_mode=args.augmentation_mode
        )
        logger.info(f"🎯 Configuration personnalisée créée")
    
    # Valider configuration
    config_manager.validate_config(config)
    
    # Sauvegarder
    config_manager.save_config(config, args.output)
    
    # Afficher commande d'entraînement
    command = config_manager.create_training_command(config)
    logger.info(f"\n🚀 Commande pour démarrer l'entraînement:")
    logger.info(command)


if __name__ == "__main__":
    main()
