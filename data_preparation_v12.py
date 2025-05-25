"""
YOLOv12-Face - Préparation des Données
Conversion WiderFace vers format YOLO avec optimisations spécifiques
"""

import os
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import shutil
import json

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WiderFaceConverter:
    """Convertisseur WiderFace vers format YOLO optimisé pour YOLOv12-Face"""
    
    def __init__(self, widerface_root, output_root):
        self.widerface_root = Path(widerface_root)
        self.output_root = Path(output_root)
        self.stats = {
            'total_images': 0,
            'total_faces': 0,
            'skipped_images': 0,
            'invalid_faces': 0,
            'size_distribution': {
                'tiny': 0,     # < 16px
                'small': 0,    # 16-32px
                'medium': 0,   # 32-64px
                'large': 0     # > 64px
            }
        }
        
    def convert_all(self):
        """Convertit train et val sets"""
        logger.info("🔄 Démarrage conversion WiderFace → YOLO")
        
        # Créer structure de sortie
        self._create_output_structure()
        
        # Convertir train et val
        for split in ['train', 'val']:
            logger.info(f"Conversion {split} set...")
            self._convert_split(split)
        
        # Créer dataset.yaml
        self._create_dataset_yaml()
        
        # Afficher statistiques
        self._print_statistics()
        
        logger.info("✅ Conversion terminée")
    
    def _create_output_structure(self):
        """Crée la structure de dossiers YOLO"""
        for split in ['train', 'val']:
            (self.output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def _convert_split(self, split):
        """Convertit un split (train/val)"""
        
        # Chemins
        ann_file = self.widerface_root / f'wider_face_split/wider_face_{split}_bbx_gt.txt'
        image_root = self.widerface_root / f'WIDER_{split}/images'
        
        if not ann_file.exists():
            logger.error(f"❌ Fichier annotations non trouvé: {ann_file}")
            return
        
        # Lire annotations
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        
        # Parser annotations
        i = 0
        pbar = tqdm(total=len(lines), desc=f"Conversion {split}")
        
        while i < len(lines):
            # Nom de l'image
            img_name = lines[i].strip()
            if not img_name:
                i += 1
                continue
                
            img_path = image_root / img_name
            
            # Nombre de visages
            i += 1
            if i >= len(lines):
                break
                
            try:
                num_faces = int(lines[i].strip())
            except ValueError:
                logger.warning(f"⚠️ Nombre de visages invalide pour {img_name}")
                i += 1
                continue
            
            # Traiter l'image
            self._process_image(img_path, img_name, lines, i+1, num_faces, split)
            
            # Avancer dans les annotations
            i += num_faces + 1
            pbar.update(num_faces + 2)
        
        pbar.close()
    
    def _process_image(self, img_path, img_name, lines, start_idx, num_faces, split):
        """Traite une image et ses annotations"""
        
        # Vérifier existence image
        if not img_path.exists():
            logger.warning(f"⚠️ Image non trouvée: {img_path}")
            self.stats['skipped_images'] += 1
            return
        
        # Lire image
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"⚠️ Impossible de lire: {img_path}")
                self.stats['skipped_images'] += 1
                return
                
            h, w = image.shape[:2]
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lecture {img_path}: {e}")
            self.stats['skipped_images'] += 1
            return
        
        # Traiter les annotations de visages
        valid_faces = []
        
        for i in range(num_faces):
            if start_idx + i >= len(lines):
                break
                
            bbox_line = lines[start_idx + i].strip().split()
            
            # Parser bbox
            if len(bbox_line) < 4:
                self.stats['invalid_faces'] += 1
                continue
            
            try:
                x, y, width, height = map(int, bbox_line[:4])
            except ValueError:
                self.stats['invalid_faces'] += 1
                continue
            
            # Valider bbox
            if self._is_valid_face(x, y, width, height, w, h):
                # Convertir en format YOLO
                yolo_bbox = self._convert_to_yolo_format(x, y, width, height, w, h)
                valid_faces.append(yolo_bbox)
                
                # Statistiques taille
                face_size = min(width, height)
                self._update_size_stats(face_size)
                
                self.stats['total_faces'] += 1
            else:
                self.stats['invalid_faces'] += 1
        
        # Sauvegarder si visages valides
        if valid_faces:
            self._save_image_and_labels(image, img_name, valid_faces, split)
            self.stats['total_images'] += 1
        else:
            self.stats['skipped_images'] += 1
    
    def _is_valid_face(self, x, y, width, height, img_w, img_h):
        """Vérifie si un visage est valide"""
        
        # Vérifier dimensions positives
        if width <= 0 or height <= 0:
            return False
        
        # Vérifier limites image
        if x < 0 or y < 0 or x + width > img_w or y + height > img_h:
            # Clip aux limites
            x = max(0, x)
            y = max(0, y)
            width = min(width, img_w - x)
            height = min(height, img_h - y)
            
            if width <= 0 or height <= 0:
                return False
        
        # Vérifier taille minimale (au moins 8px)
        if min(width, height) < 8:
            return False
        
        # Vérifier ratio aspect (éviter boîtes trop déformées)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 5.0:  # Ratio maximum 5:1
            return False
        
        return True
    
    def _convert_to_yolo_format(self, x, y, width, height, img_w, img_h):
        """Convertit bbox au format YOLO normalisé"""
        
        # Centre de la boîte
        center_x = (x + width / 2) / img_w
        center_y = (y + height / 2) / img_h
        
        # Dimensions normalisées
        norm_width = width / img_w
        norm_height = height / img_h
        
        return [0, center_x, center_y, norm_width, norm_height]  # Classe 0 = face
    
    def _update_size_stats(self, face_size):
        """Met à jour les statistiques de taille"""
        if face_size < 16:
            self.stats['size_distribution']['tiny'] += 1
        elif face_size < 32:
            self.stats['size_distribution']['small'] += 1
        elif face_size < 64:
            self.stats['size_distribution']['medium'] += 1
        else:
            self.stats['size_distribution']['large'] += 1
    
    def _save_image_and_labels(self, image, img_name, faces, split):
        """Sauvegarde image et labels"""
        
        # Chemins de sortie
        img_out_path = self.output_root / 'images' / split / Path(img_name).name
        label_out_path = self.output_root / 'labels' / split / f"{Path(img_name).stem}.txt"
        
        # Créer dossiers si nécessaire
        img_out_path.parent.mkdir(parents=True, exist_ok=True)
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder image
        cv2.imwrite(str(img_out_path), image)
        
        # Sauvegarder labels
        with open(label_out_path, 'w') as f:
            for face in faces:
                f.write(f"{face[0]} {face[1]:.6f} {face[2]:.6f} {face[3]:.6f} {face[4]:.6f}\n")
    
    def _create_dataset_yaml(self):
        """Crée le fichier dataset.yaml pour YOLOv12"""
        
        dataset_config = {
            'path': str(self.output_root),
            'train': 'images/train',
            'val': 'images/val',
            
            # Classes
            'nc': 1,
            'names': ['face'],
            
            # Métadonnées
            'description': 'WiderFace dataset convertit pour YOLOv12-Face',
            'version': '1.0',
            'license': 'Creative Commons',
            'url': 'http://shuoyang1213.me/WIDERFACE/',
            
            # Configuration YOLOv12 spécifique
            'yolov12_config': {
                'attention_type': 'area',
                'backbone': 'yolov12',
                'neck': 'r_elan',
                'optimized_for': 'face_detection'
            },
            
            # Statistiques dataset
            'statistics': {
                'total_images': self.stats['total_images'],
                'total_annotations': self.stats['total_faces'],
                'size_distribution': self.stats['size_distribution']
            }
        }
        
        yaml_path = self.output_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset YAML créé: {yaml_path}")
    
    def _print_statistics(self):
        """Affiche les statistiques de conversion"""
        
        logger.info("\n" + "="*60)
        logger.info("📊 STATISTIQUES DE CONVERSION")
        logger.info("="*60)
        
        logger.info(f"Images traitées: {self.stats['total_images']}")
        logger.info(f"Images ignorées: {self.stats['skipped_images']}")
        logger.info(f"Visages valides: {self.stats['total_faces']}")
        logger.info(f"Visages invalides: {self.stats['invalid_faces']}")
        
        logger.info("\nDistribution par taille:")
        total_faces = self.stats['total_faces']
        if total_faces > 0:
            for size_cat, count in self.stats['size_distribution'].items():
                percentage = (count / total_faces) * 100
                logger.info(f"  {size_cat.capitalize()}: {count} ({percentage:.1f}%)")
        
        logger.info("="*60)


class DataAugmentationAnalyzer:
    """Analyseur pour optimiser l'augmentation des données"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.train_images = list((self.dataset_path / 'images/train').glob('*.jpg'))
        self.train_labels = list((self.dataset_path / 'labels/train').glob('*.txt'))
    
    def analyze_face_characteristics(self):
        """Analyse les caractéristiques des visages pour optimiser l'augmentation"""
        
        logger.info("🔍 Analyse caractéristiques des visages...")
        
        face_sizes = []
        aspect_ratios = []
        positions = []
        
        for label_file in tqdm(self.train_labels[:1000], desc="Analyse"):  # Échantillon
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, w, h = map(float, parts[:5])
                        
                        face_sizes.append(min(w, h))
                        aspect_ratios.append(max(w, h) / min(w, h))
                        positions.append((cx, cy))
                        
            except Exception as e:
                continue
        
        # Statistiques
        if face_sizes:
            logger.info(f"Taille moyenne des visages: {np.mean(face_sizes):.3f}")
            logger.info(f"Ratio aspect moyen: {np.mean(aspect_ratios):.3f}")
            logger.info(f"Position moyenne: ({np.mean([p[0] for p in positions]):.3f}, {np.mean([p[1] for p in positions]):.3f})")
            
            # Recommandations augmentation
            self._generate_augmentation_recommendations(face_sizes, aspect_ratios)
    
    def _generate_augmentation_recommendations(self, face_sizes, aspect_ratios):
        """Génère des recommandations d'augmentation"""
        
        logger.info("\n💡 Recommandations d'augmentation:")
        
        # Analyse tailles
        small_faces_ratio = sum(1 for s in face_sizes if s < 0.05) / len(face_sizes)
        if small_faces_ratio > 0.3:
            logger.info("  - Augmenter mosaic probability (beaucoup de petits visages)")
            logger.info("  - Réduire scale augmentation")
        
        # Analyse ratios
        avg_ratio = np.mean(aspect_ratios)
        if avg_ratio > 1.5:
            logger.info("  - Limiter perspective augmentation")
            logger.info("  - Éviter shear transformation")
        
        logger.info("  - Pas de rotation (visages sensibles à l'orientation)")
        logger.info("  - Flip horizontal OK, éviter flip vertical")


def create_custom_splits(dataset_path, train_ratio=0.8, val_ratio=0.2):
    """Crée des splits personnalisés du dataset"""
    
    logger.info("✂️ Création splits personnalisés...")
    
    dataset_path = Path(dataset_path)
    
    # Lister toutes les images
    all_images = list((dataset_path / 'images/train').glob('*.jpg'))
    all_images.extend(list((dataset_path / 'images/val').glob('*.jpg')))
    
    # Mélanger
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    # Calculer splits
    total_images = len(all_images)
    train_split = int(total_images * train_ratio)
    
    train_images = all_images[:train_split]
    val_images = all_images[train_split:]
    
    logger.info(f"Train: {len(train_images)} images")
    logger.info(f"Val: {len(val_images)} images")
    
    # Réorganiser fichiers (optionnel - peut être fait si nécessaire)
    return train_images, val_images


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Préparation données YOLOv12-Face')
    parser.add_argument('--widerface-root', type=str, default='/content/datasets/widerface',
                       help='Chemin root WiderFace')
    parser.add_argument('--output-root', type=str, default='/content/datasets/yolo_widerface',
                       help='Chemin sortie YOLO')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Analyser dataset existant uniquement')
    parser.add_argument('--convert-only', action='store_true',
                       help='Conversion uniquement')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Analyse uniquement
        analyzer = DataAugmentationAnalyzer(args.output_root)
        analyzer.analyze_face_characteristics()
    elif args.convert_only:
        # Conversion uniquement
        converter = WiderFaceConverter(args.widerface_root, args.output_root)
        converter.convert_all()
    else:
        # Conversion + analyse
        converter = WiderFaceConverter(args.widerface_root, args.output_root)
        converter.convert_all()
        
        analyzer = DataAugmentationAnalyzer(args.output_root)
        analyzer.analyze_face_characteristics()


if __name__ == "__main__":
    main()
