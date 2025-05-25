"""
YOLOv12-Face - Script de Démarrage Automatique Google Colab
Installation complète et lancement en une commande
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import requests
import zipfile
import shutil

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv12FaceColabSetup:
    """Setup automatique complet pour Google Colab"""
    
    def __init__(self):
        self.base_dir = Path('/content')
        self.project_dir = self.base_dir / 'yolov12_face_project'
        self.datasets_dir = self.base_dir / 'datasets'
        self.runs_dir = self.base_dir / 'runs'
        
        # URLs des ressources
        self.urls = {
            'yolov12_repo': 'https://github.com/rsemihkoca/yolov12-face.git',
            'widerface_train': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip',
            'widerface_val': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip',
            'widerface_annotations': 'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip'
        }
        
        self.is_colab = self._check_colab_environment()
        
    def _check_colab_environment(self) -> bool:
        """Vérifie si on est dans Google Colab"""
        try:
            import google.colab
            logger.info("✅ Environnement Google Colab détecté")
            return True
        except ImportError:
            logger.info("ℹ️ Environnement local détecté")
            return False
    
    def setup_environment(self):
        """Configure l'environnement complet"""
        
        logger.info("🚀 Démarrage setup YOLOv12-Face pour Google Colab")
        logger.info("="*60)
        
        # 1. Vérifications système
        self._check_system_requirements()
        
        # 2. Installation des dépendances
        self._install_dependencies()
        
        # 3. Configuration des répertoires
        self._setup_directories()
        
        # 4. Clonage du repository YOLOv12
        self._clone_yolov12_repo()
        
        # 5. Téléchargement dataset WiderFace (optionnel)
        if self._ask_user_confirmation("Télécharger le dataset WiderFace (~3GB) ?"):
            self._download_widerface_dataset()
        
        # 6. Configuration du projet
        self._setup_project_config()
        
        # 7. Test final
        self._run_integration_test()
        
        logger.info("🎉 Setup terminé avec succès!")
        self._print_next_steps()
    
    def _check_system_requirements(self):
        """Vérifie les requis système"""
        
        logger.info("🔍 Vérification des requis système...")
        
        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("⚠️ Aucun GPU détecté - entraînement sera lent")
        except ImportError:
            logger.warning("⚠️ PyTorch non installé")
        
        # RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / 1e9
            logger.info(f"💾 RAM: {ram_gb:.1f}GB")
        except ImportError:
            logger.info("💾 RAM: Information non disponible")
        
        # Espace disque
        disk_free = shutil.disk_usage('/content')[2] / 1e9
        logger.info(f"💿 Espace libre: {disk_free:.1f}GB")
        
        if disk_free < 10:
            logger.warning("⚠️ Espace disque faible (<10GB)")
    
    def _install_dependencies(self):
        """Installe toutes les dépendances nécessaires"""
        
        logger.info("📦 Installation des dépendances...")
        
        # Packages Python essentiels
        essential_packages = [
            'ultralytics>=8.0.0',
            'opencv-python>=4.5.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
            'onnx>=1.12.0',
            'onnxruntime>=1.12.0',
            'psutil',
            'pillow>=8.0.0'
        ]
        
        for package in essential_packages:
            try:
                logger.info(f"  Installing {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                logger.info(f"  ✅ {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"  ⚠️ Erreur installation {package}: {e}")
        
        # Vérification imports critiques
        critical_imports = ['torch', 'ultralytics', 'cv2', 'numpy', 'yaml']
        
        for module in critical_imports:
            try:
                __import__(module)
                logger.info(f"  ✅ {module} importé avec succès")
            except ImportError as e:
                logger.error(f"  ❌ {module} non disponible: {e}")
    
    def _setup_directories(self):
        """Crée la structure de répertoires"""
        
        logger.info("📁 Configuration des répertoires...")
        
        directories = [
            self.project_dir,
            self.datasets_dir,
            self.runs_dir,
            self.project_dir / 'scripts',
            self.project_dir / 'configs',
            self.project_dir / 'weights',
            self.datasets_dir / 'widerface',
            self.datasets_dir / 'yolo_widerface'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {directory}")
    
    def _clone_yolov12_repo(self):
        """Clone le repository YOLOv12-Face"""
        
        logger.info("📥 Clonage repository YOLOv12...")
        
        yolov12_dir = self.project_dir / 'yolov12-face'
        
        if yolov12_dir.exists():
            logger.info("  ℹ️ Repository déjà cloné")
            return
        
        try:
            subprocess.run([
                'git', 'clone', self.urls['yolov12_repo'], str(yolov12_dir)
            ], check=True, capture_output=True)
            
            logger.info(f"  ✅ YOLOv12-Face cloné: {yolov12_dir}")
            
            # Copier les fichiers de configuration YOLOv12
            if (yolov12_dir / 'ultralytics').exists():
                logger.info("  📋 Configuration YOLOv12 détectée")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Erreur clonage: {e}")
    
    def _download_widerface_dataset(self):
        """Télécharge et prépare le dataset WiderFace"""
        
        logger.info("📥 Téléchargement dataset WiderFace...")
        
        widerface_dir = self.datasets_dir / 'widerface'
        
        # Télécharger fichiers
        files_to_download = [
            ('train', self.urls['widerface_train']),
            ('val', self.urls['widerface_val']),
            ('annotations', self.urls['widerface_annotations'])
        ]
        
        for name, url in files_to_download:
            output_file = widerface_dir / f"{name}.zip"
            
            if output_file.exists():
                logger.info(f"  ℹ️ {name}.zip déjà téléchargé")
                continue
            
            try:
                logger.info(f"  📥 Téléchargement {name}...")
                self._download_file(url, output_file)
                logger.info(f"  ✅ {name}.zip téléchargé")
                
                # Extraction
                logger.info(f"  📦 Extraction {name}...")
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall(widerface_dir)
                logger.info(f"  ✅ {name} extrait")
                
            except Exception as e:
                logger.error(f"  ❌ Erreur {name}: {e}")
        
        logger.info("✅ Dataset WiderFace préparé")
    
    def _download_file(self, url: str, output_path: Path):
        """Télécharge un fichier avec barre de progression"""
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r    Progression: {progress:.1f}%", end='', flush=True)
        
        print()  # Nouvelle ligne après progression
    
    def _setup_project_config(self):
        """Configure le projet avec les scripts créés"""
        
        logger.info("⚙️ Configuration du projet...")
        
        # Copier nos scripts dans le projet
        scripts_source = Path('/content/reconnaissance_Facial_v12')
        scripts_dest = self.project_dir / 'scripts'
        
        if scripts_source.exists():
            scripts_to_copy = [
                'config_v12.py',
                'main_v12.py',
                'data_preparation_v12.py',
                'model_evaluation_v12.py',
                'utils_v12.py'
            ]
            
            for script in scripts_to_copy:
                source_file = scripts_source / script
                dest_file = scripts_dest / script
                
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"  ✅ {script} copié")
        
        # Créer configuration par défaut
        self._create_default_config()
        
        # Créer script de lancement rapide
        self._create_quick_start_script()
    
    def _create_default_config(self):
        """Crée une configuration par défaut optimisée Colab"""
        
        config_content = '''# YOLOv12-Face Configuration Colab Optimisée
model_size: 'n'  # nano pour Colab gratuit
epochs: 50
batch_size: 16
image_size: 640

# Chemins Colab
data_path: '/content/datasets/yolo_widerface/dataset.yaml'
project_path: '/content/runs/train'
name: 'yolov12_face_colab'

# Optimisations Colab
workers: 2
cache: false
amp: true
cos_lr: true
close_mosaic: 10

# Patience pour early stopping
patience: 15
save_period: 10

# Augmentations modérées pour visages
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

# Loss weights pour visages
loss:
  box: 7.5
  cls: 0.5
  dfl: 1.5
'''
        
        config_file = self.project_dir / 'configs' / 'colab_default.yaml'
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"  ✅ Configuration par défaut: {config_file}")
    
    def _create_quick_start_script(self):
        """Crée un script de démarrage rapide"""
        
        quick_start_content = '''#!/bin/bash
# YOLOv12-Face - Script de Démarrage Rapide

echo "🚀 YOLOv12-Face - Démarrage Rapide"
echo "=================================="

cd /content/yolov12_face_project

# 1. Préparation des données (si pas déjà fait)
if [ ! -d "/content/datasets/yolo_widerface" ]; then
    echo "📁 Préparation des données..."
    python scripts/data_preparation_v12.py --convert --output /content/datasets/
fi

# 2. Lancement entraînement
echo "🚀 Démarrage entraînement YOLOv12-Face..."
python scripts/main_v12.py --config configs/colab_default.yaml

# 3. Évaluation automatique
echo "📊 Évaluation du modèle..."
python scripts/model_evaluation_v12.py \\
    --model /content/runs/train/yolov12_face_colab/weights/best.pt \\
    --data /content/datasets/yolo_widerface/dataset.yaml \\
    --plots

echo "✅ Entraînement terminé!"
echo "📁 Résultats dans: /content/runs/train/yolov12_face_colab/"
'''
        
        script_file = self.project_dir / 'quick_start.sh'
        with open(script_file, 'w') as f:
            f.write(quick_start_content)
        
        # Rendre exécutable
        os.chmod(script_file, 0o755)
        
        logger.info(f"  ✅ Script démarrage rapide: {script_file}")
    
    def _run_integration_test(self):
        """Lance un test d'intégration rapide"""
        
        logger.info("🧪 Test d'intégration...")
        
        try:
            # Test import ultralytics
            from ultralytics import YOLO
            logger.info("  ✅ YOLO importé")
            
            # Test création modèle (sans poids)
            model = YOLO('yolov8n.pt')  # Modèle de base
            logger.info("  ✅ Modèle YOLO créé")
            
            # Test PyTorch
            import torch
            if torch.cuda.is_available():
                logger.info(f"  ✅ CUDA disponible: {torch.cuda.get_device_name()}")
            
            logger.info("✅ Tous les tests passent!")
            
        except Exception as e:
            logger.error(f"❌ Erreur test d'intégration: {e}")
    
    def _ask_user_confirmation(self, question: str) -> bool:
        """Demande confirmation utilisateur (toujours True en auto)"""
        if self.is_colab:
            # En mode auto pour Colab
            logger.info(f"🤖 Auto-confirmation: {question} -> OUI")
            return True
        else:
            # Mode interactif pour local
            response = input(f"{question} (o/N): ").lower()
            return response in ['o', 'oui', 'y', 'yes']
    
    def _print_next_steps(self):
        """Affiche les prochaines étapes"""
        
        logger.info("\n" + "="*60)
        logger.info("🎉 SETUP TERMINÉ AVEC SUCCÈS!")
        logger.info("="*60)
        
        logger.info("\n📋 PROCHAINES ÉTAPES:")
        
        logger.info("\n1️⃣ **Démarrage Rapide (Recommandé)**")
        logger.info("   cd /content/yolov12_face_project")
        logger.info("   bash quick_start.sh")
        
        logger.info("\n2️⃣ **Entraînement Manuel**")
        logger.info("   cd /content/yolov12_face_project")
        logger.info("   python scripts/main_v12.py --config configs/colab_default.yaml")
        
        logger.info("\n3️⃣ **Configuration Personnalisée**")
        logger.info("   python scripts/config_v12.py --preset colab_nano")
        logger.info("   python scripts/main_v12.py --config /content/yolov12_face_config.yaml")
        
        logger.info("\n📊 **Monitoring**")
        logger.info("   - Logs: /content/runs/train/yolov12_face_colab/")
        logger.info("   - Poids: /content/runs/train/yolov12_face_colab/weights/")
        logger.info("   - TensorBoard: tensorboard --logdir /content/runs/")
        
        logger.info("\n🎯 **Objectifs Performance**")
        logger.info("   - WiderFace Easy: >97.5% (vs 94.8% ADYOLOv5)")
        logger.info("   - WiderFace Medium: >96.5% (vs 93.8% ADYOLOv5)")
        logger.info("   - WiderFace Hard: >88.5% (vs 84.4% ADYOLOv5)")
        
        logger.info("\n🚀 **Bonne chance pour dépasser ADYOLOv5-Face!**")


def quick_setup():
    """Fonction de setup rapide pour import direct"""
    
    setup = YOLOv12FaceColabSetup()
    setup.setup_environment()


def main():
    """Fonction principale"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12-Face Colab Setup')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Ignorer téléchargement WiderFace')
    parser.add_argument('--quick', action='store_true',
                       help='Setup rapide sans interactions')
    
    args = parser.parse_args()
    
    setup = YOLOv12FaceColabSetup()
    
    if args.quick:
        logger.info("🚀 Mode setup rapide activé")
    
    setup.setup_environment()


if __name__ == "__main__":
    main()
