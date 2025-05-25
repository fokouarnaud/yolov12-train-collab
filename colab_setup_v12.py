"""
YOLOv12-Face - Configuration et Setup pour Google Colab
Installation et préparation de l'environnement YOLOv12-Face
"""

import os
import sys
import subprocess
import logging
import yaml
import urllib.request
import zipfile
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv12FaceSetup:
    """Classe pour setup complet YOLOv12-Face sur Colab"""
    
    def __init__(self):
        self.base_dir = "/content"
        self.workspace_dir = f"{self.base_dir}/yolov12_workspace"
        self.dataset_dir = f"{self.base_dir}/datasets"
        self.yolo_dir = f"{self.base_dir}/yolov12-face"
        
    def check_environment(self):
        """Vérifie l'environnement Colab"""
        logger.info("🔍 Vérification environnement...")
        
        # Vérifier GPU
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU disponible: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️ GPU non disponible, utilisation CPU")
        
        # Vérifier RAM
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        logger.info(f"💾 RAM disponible: {ram_gb:.1f}GB")
        
        # Vérifier espace disque
        disk_gb = psutil.disk_usage('/').free / 1e9
        logger.info(f"💿 Espace disque libre: {disk_gb:.1f}GB")
        
        if disk_gb < 15:
            logger.warning("⚠️ Espace disque faible, nettoyage recommandé")
    
    def install_dependencies(self):
        """Installation des dépendances YOLOv12-Face"""
        logger.info("📦 Installation des dépendances...")
        
        # Mise à jour pip
        self._run_command("pip install --upgrade pip")
        
        # Désinstaller ultralytics existant pour éviter conflits
        self._run_command("pip uninstall -y ultralytics", ignore_errors=True)
        
        # Installer PyTorch compatible
        pytorch_cmd = (
            "pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        self._run_command(pytorch_cmd)
        
        # Dépendances principales
        dependencies = [
            "numpy>=1.24.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.5.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
            "tqdm>=4.65.0",
            "PyYAML>=6.0",
            "requests>=2.31.0",
            "psutil>=5.9.0",
            "py-cpuinfo>=9.0.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0"
        ]
        
        for dep in dependencies:
            self._run_command(f"pip install {dep}")
        
        # Installer FlashAttention (optionnel)
        self._install_flash_attention()
        
        logger.info("✅ Dépendances installées")
    
    def _install_flash_attention(self):
        """Installation FlashAttention pour optimisation"""
        logger.info("⚡ Installation FlashAttention...")
        
        try:
            # Vérifier compatibilité GPU
            import torch
            if not torch.cuda.is_available():
                logger.warning("⚠️ GPU non disponible, FlashAttention ignoré")
                return
            
            # Installer FlashAttention
            flash_cmd = "pip install flash-attn --no-build-isolation"
            result = self._run_command(flash_cmd, ignore_errors=True)
            
            if result == 0:
                logger.info("✅ FlashAttention installé")
            else:
                logger.warning("⚠️ FlashAttention non installé (GPU incompatible?)")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur FlashAttention: {e}")
    
    def clone_yolov12_face(self):
        """Clone le repository YOLOv12-Face"""
        logger.info("📥 Clonage YOLOv12-Face...")
        
        # Supprimer dossier existant
        if os.path.exists(self.yolo_dir):
            self._run_command(f"rm -rf {self.yolo_dir}")
        
        # Clone depuis votre fork
        clone_cmd = f"git clone https://github.com/fokouarnaud/yolov12-face.git {self.yolo_dir}"
        self._run_command(clone_cmd)
        
        # Installation en mode développement
        os.chdir(self.yolo_dir)
        self._run_command("pip install -e .")
        
        # Retour répertoire base
        os.chdir(self.base_dir)
        
        logger.info("✅ YOLOv12-Face cloné et installé")
    
    def verify_installation(self):
        """Vérifie l'installation YOLOv12-Face"""
        logger.info("🔍 Vérification installation...")
        
        try:
            from ultralytics import YOLO
            
            # Test création modèle
            model = YOLO('yolov12n.yaml')
            logger.info("✅ YOLOv12 chargé avec succès")
            
            # Vérifier FlashAttention
            try:
                import flash_attn
                logger.info("✅ FlashAttention disponible")
            except ImportError:
                logger.info("ℹ️ FlashAttention non disponible (optionnel)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification: {e}")
            return False
    
    def setup_workspace(self):
        """Crée la structure de workspace"""
        logger.info("📁 Création workspace...")
        
        directories = [
            self.workspace_dir,
            f"{self.dataset_dir}/widerface",
            f"{self.dataset_dir}/yolo_widerface/images/train",
            f"{self.dataset_dir}/yolo_widerface/images/val",
            f"{self.dataset_dir}/yolo_widerface/labels/train",
            f"{self.dataset_dir}/yolo_widerface/labels/val",
            f"{self.base_dir}/runs/train",
            f"{self.base_dir}/runs/val",
            f"{self.base_dir}/runs/export"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("✅ Workspace créé")
    
    def download_widerface(self):
        """Télécharge le dataset WiderFace"""
        logger.info("📊 Téléchargement WiderFace dataset...")
        
        widerface_dir = f"{self.dataset_dir}/widerface"
        
        # URLs WiderFace
        urls = {
            'WIDER_train.zip': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip',
            'WIDER_val.zip': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip',
            'wider_face_split.zip': 'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip'
        }
        
        for filename, url in urls.items():
            file_path = f"{widerface_dir}/{filename}"
            
            if not os.path.exists(file_path):
                logger.info(f"  Téléchargement {filename}...")
                self._download_file(url, file_path)
            else:
                logger.info(f"  {filename} déjà téléchargé")
        
        # Extraction
        for filename in urls.keys():
            file_path = f"{widerface_dir}/{filename}"
            extract_dir = widerface_dir
            
            logger.info(f"  Extraction {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        logger.info("✅ WiderFace téléchargé et extrait")
    
    def convert_widerface_to_yolo(self):
        """Convertit WiderFace au format YOLO"""
        logger.info("🔄 Conversion WiderFace → YOLO...")
        
        from data_preparation_v12 import WiderFaceConverter
        
        converter = WiderFaceConverter(
            widerface_root=f"{self.dataset_dir}/widerface",
            output_root=f"{self.dataset_dir}/yolo_widerface"
        )
        
        converter.convert_all()
        
        logger.info("✅ Conversion terminée")
    
    def create_config_files(self):
        """Crée les fichiers de configuration"""
        logger.info("⚙️ Création fichiers configuration...")
        
        # Configuration dataset YAML
        dataset_config = {
            'path': f"{self.dataset_dir}/yolo_widerface",
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['face'],
            
            # Configuration YOLOv12 spécifique
            'attention_type': 'area',
            'backbone': 'yolov12',
            'neck': 'r_elan'
        }
        
        dataset_yaml_path = f"{self.dataset_dir}/yolo_widerface/dataset.yaml"
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Configuration entraînement
        training_config = {
            'model_size': 'n',
            'epochs': 100,
            'batch_size': 16,
            'image_size': 640,
            'data_path': dataset_yaml_path,
            'project_path': f"{self.base_dir}/runs/train",
            'name': 'yolov12_face',
            'patience': 20,
            'save_period': 10,
            'workers': 2,
            'cache': False,
            'amp': True,
            'cos_lr': True,
            'close_mosaic': 10,
            
            'augmentation': {
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
                'copy_paste': 0.0
            },
            
            'loss': {
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5
            },
            
            'yolov12': {
                'attention_type': 'area',
                'num_regions': 4,
                'flash_attention': True,
                'r_elan': True,
                'residual_scaling': True
            }
        }
        
        config_path = f"{self.base_dir}/yolov12_face_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        logger.info("✅ Fichiers configuration créés")
        return config_path
    
    def _run_command(self, command, ignore_errors=False):
        """Exécute une commande shell"""
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            return 0
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                logger.error(f"Erreur commande: {command}")
                logger.error(f"Sortie erreur: {e.stderr}")
            return e.returncode
        except Exception as e:
            if not ignore_errors:
                logger.error(f"Erreur inattendue: {e}")
            return 1
    
    def _download_file(self, url, filepath):
        """Télécharge un fichier avec barre de progression"""
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                logger.info(f"    Progression: {percent}%")
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
    
    def full_setup(self, model_size='n'):
        """Setup complet YOLOv12-Face"""
        logger.info("🚀 Démarrage setup complet YOLOv12-Face")
        
        steps = [
            ("Vérification environnement", self.check_environment),
            ("Installation dépendances", self.install_dependencies),
            ("Setup workspace", self.setup_workspace),
            ("Clonage YOLOv12-Face", self.clone_yolov12_face),
            ("Vérification installation", self.verify_installation),
            ("Téléchargement WiderFace", self.download_widerface),
            ("Conversion dataset", self.convert_widerface_to_yolo),
            ("Création configurations", self.create_config_files)
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"ÉTAPE: {step_name}")
                logger.info(f"{'='*60}")
                
                step_func()
                
                logger.info(f"✅ {step_name} terminé")
                
            except Exception as e:
                logger.error(f"❌ Erreur dans {step_name}: {e}")
                raise
        
        logger.info("\n🎉 Setup YOLOv12-Face terminé avec succès!")
        logger.info("\nPour commencer l'entraînement:")
        logger.info("python main_v12.py --model-size n")
        
        return True


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup YOLOv12-Face sur Colab')
    parser.add_argument('--model-size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Taille du modèle YOLOv12')
    parser.add_argument('--skip-download', action='store_true',
                       help='Ignorer téléchargement dataset')
    parser.add_argument('--skip-conversion', action='store_true',
                       help='Ignorer conversion dataset')
    
    args = parser.parse_args()
    
    # Créer instance setup
    setup = YOLOv12FaceSetup()
    
    try:
        if args.skip_download and args.skip_conversion:
            # Setup minimal
            setup.check_environment()
            setup.install_dependencies()
            setup.setup_workspace()
            setup.clone_yolov12_face()
            setup.verify_installation()
            setup.create_config_files()
        else:
            # Setup complet
            setup.full_setup(args.model_size)
        
        logger.info("✅ Setup terminé avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
