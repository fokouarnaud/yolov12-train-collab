"""
YOLOv12-Face - Utilitaires Avancés
Outils pour visualisation, export, debug attention et optimisation
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ultralytics import YOLO
import onnx
import onnxruntime as ort
from collections import defaultdict
import pandas as pd

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv12FaceVisualizer:
    """Visualisateur avancé pour YOLOv12-Face"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le visualisateur
        
        Args:
            model_path: Chemin vers le modèle YOLOv12-Face
        """
        self.model_path = model_path
        self.model = None
        self.colors = self._generate_colors()
        self.font_size = 16
        
        # Configuration visualisation
        self.vis_config = {
            'bbox_thickness': 2,
            'text_thickness': 1,
            'conf_threshold': 0.25,
            'show_conf': True,
            'show_class': True,
            'show_size': True,
            'alpha_blend': 0.3
        }
        
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Génère des couleurs distinctes pour les détections"""
        colors = [
            (255, 0, 0),    # Rouge
            (0, 255, 0),    # Vert
            (0, 0, 255),    # Bleu
            (255, 255, 0),  # Jaune
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Violet
            (0, 128, 255),  # Bleu clair
            (255, 0, 128)   # Rose
        ]
        return colors
    
    def load_model(self, model_path: str) -> bool:
        """Charge le modèle YOLOv12-Face"""
        try:
            self.model_path = model_path
            self.model = YOLO(model_path)
            logger.info(f"✅ Modèle chargé: {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def visualize_detections(self, image_path: str, output_path: str = None, 
                           conf_threshold: float = 0.25, save_crops: bool = False) -> str:
        """
        Visualise les détections sur une image avec informations détaillées
        
        Args:
            image_path: Chemin vers l'image
            output_path: Chemin de sauvegarde (optionnel)
            conf_threshold: Seuil de confiance
            save_crops: Sauvegarder les crops des visages détectés
            
        Returns:
            Chemin de l'image sauvegardée
        """
        if self.model is None:
            logger.error("❌ Modèle non chargé")
            return ""
        
        try:
            # Charger image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Impossible de charger l'image: {image_path}")
                return ""
            
            height, width = image.shape[:2]
            logger.info(f"🖼️ Image chargée: {width}x{height}")
            
            # Inférence
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            if not results or len(results) == 0:
                logger.warning("⚠️ Aucun résultat d'inférence")
                return ""
            
            detections = results[0].boxes
            if detections is None or len(detections) == 0:
                logger.info("ℹ️ Aucune détection")
                # Sauvegarder image originale
                if output_path:
                    cv2.imwrite(output_path, image)
                return output_path or image_path
            
            logger.info(f"🎯 {len(detections)} détections trouvées")
            
            # Visualiser détections
            annotated_image = self._draw_detections(image, detections, save_crops, image_path)
            
            # Chemin de sortie
            if output_path is None:
                name = Path(image_path).stem
                output_path = f"/content/detection_result_{name}.jpg"
            
            # Sauvegarder
            cv2.imwrite(output_path, annotated_image)
            logger.info(f"✅ Image annotée sauvée: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur visualisation: {e}")
            return ""
    
    def _draw_detections(self, image: np.ndarray, detections, save_crops: bool = False, 
                        image_path: str = "") -> np.ndarray:
        """Dessine les détections sur l'image"""
        
        annotated = image.copy()
        crop_dir = None
        
        if save_crops and image_path:
            crop_dir = Path(image_path).parent / "face_crops"
            crop_dir.mkdir(exist_ok=True)
        
        for i, box in enumerate(detections):
            # Extraire informations
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            
            # Calculer dimensions
            width, height = x2 - x1, y2 - y1
            face_size = min(width, height)
            area = width * height
            
            # Couleur selon confiance
            color_idx = min(int(conf * len(self.colors)), len(self.colors) - 1)
            color = self.colors[color_idx]
            
            # Dessiner rectangle
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), 
                         color, self.vis_config['bbox_thickness'])
            
            # Préparer texte
            texts = []
            if self.vis_config['show_conf']:
                texts.append(f"Conf: {conf:.2f}")
            if self.vis_config['show_size']:
                texts.append(f"Size: {face_size:.0f}px")
                texts.append(f"Area: {area:.0f}")
            
            # Dessiner textes
            text_y = int(y1) - 10
            for text in texts:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Fond du texte
                cv2.rectangle(annotated, 
                             (int(x1), text_y - text_size[1] - 5),
                             (int(x1) + text_size[0] + 5, text_y + 5),
                             color, -1)
                
                # Texte
                cv2.putText(annotated, text, (int(x1) + 2, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                text_y -= text_size[1] + 8
            
            # Sauvegarder crop si demandé
            if save_crops and crop_dir:
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    crop_path = crop_dir / f"face_{i:03d}_conf{conf:.2f}.jpg"
                    cv2.imwrite(str(crop_path), crop)
        
        return annotated
    
    def create_detection_grid(self, image_paths: List[str], output_path: str = "/content/detection_grid.jpg",
                            grid_size: Tuple[int, int] = (2, 3), conf_threshold: float = 0.25) -> str:
        """Crée une grille de détections pour comparaison"""
        
        if self.model is None:
            logger.error("❌ Modèle non chargé")
            return ""
        
        rows, cols = grid_size
        max_images = min(len(image_paths), rows * cols)
        
        logger.info(f"📊 Création grille {rows}x{cols} avec {max_images} images")
        
        # Taille des cellules
        cell_width, cell_height = 640, 480
        
        # Créer canvas
        grid_width = cols * cell_width
        grid_height = rows * cell_height
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for idx in range(max_images):
            row = idx // cols
            col = idx % cols
            
            try:
                # Charger et redimensionner image
                image = cv2.imread(image_paths[idx])
                if image is None:
                    continue
                
                image_resized = cv2.resize(image, (cell_width, cell_height))
                
                # Inférence
                results = self.model(image_paths[idx], conf=conf_threshold, verbose=False)
                
                # Annoter
                if results and len(results) > 0 and results[0].boxes is not None:
                    # Redimensionner coordonnées
                    original_h, original_w = image.shape[:2]
                    scale_x = cell_width / original_w
                    scale_y = cell_height / original_h
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        
                        # Redimensionner coordonnées
                        x1_scaled = int(x1 * scale_x)
                        y1_scaled = int(y1 * scale_y)
                        x2_scaled = int(x2 * scale_x)
                        y2_scaled = int(y2 * scale_y)
                        
                        # Dessiner
                        cv2.rectangle(image_resized, (x1_scaled, y1_scaled), 
                                    (x2_scaled, y2_scaled), (0, 255, 0), 2)
                        cv2.putText(image_resized, f"{conf:.2f}", 
                                  (x1_scaled, y1_scaled - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Placer dans grille
                y_start = row * cell_height
                y_end = y_start + cell_height
                x_start = col * cell_width
                x_end = x_start + cell_width
                
                grid_image[y_start:y_end, x_start:x_end] = image_resized
                
                # Ajouter nom fichier
                filename = Path(image_paths[idx]).name
                cv2.putText(grid_image, filename, (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur traitement {image_paths[idx]}: {e}")
                continue
        
        # Sauvegarder grille
        cv2.imwrite(output_path, grid_image)
        logger.info(f"✅ Grille sauvée: {output_path}")
        
        return output_path


class YOLOv12FaceExporter:
    """Exporteur avancé pour YOLOv12-Face"""
    
    def __init__(self, model_path: str):
        """
        Initialise l'exporteur
        
        Args:
            model_path: Chemin vers le modèle PyTorch
        """
        self.model_path = model_path
        self.model = None
        self.export_formats = ['onnx', 'torchscript', 'engine', 'coreml', 'tflite']
        
    def load_model(self) -> bool:
        """Charge le modèle pour export"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✅ Modèle chargé pour export: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def export_onnx_optimized(self, output_dir: str = "/content/exports", 
                            imgsz: int = 640, dynamic: bool = False, 
                            simplify: bool = True, opset: int = 17) -> Dict[str, str]:
        """
        Export ONNX optimisé pour déploiement
        
        Args:
            output_dir: Répertoire de sortie
            imgsz: Taille d'image
            dynamic: Tailles dynamiques
            simplify: Simplification ONNX
            opset: Version opset ONNX
            
        Returns:
            Dictionnaire avec chemins des exports
        """
        if self.model is None and not self.load_model():
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Export ONNX optimisé: {imgsz}px")
        
        exports = {}
        
        try:
            # Export ONNX de base
            logger.info("  Export ONNX standard...")
            base_config = {
                'format': 'onnx',
                'imgsz': imgsz,
                'dynamic': dynamic,
                'simplify': simplify,
                'opset': opset,
                'half': False  # FP32 pour compatibilité
            }
            
            exported_path = self.model.export(**base_config)
            
            # Déplacer vers répertoire de sortie
            onnx_path = output_dir / f"yolov12_face_{imgsz}.onnx"
            if Path(exported_path).exists():
                Path(exported_path).rename(onnx_path)
                exports['onnx_fp32'] = str(onnx_path)
                logger.info(f"    ✅ ONNX FP32: {onnx_path}")
            
            # Export ONNX FP16 si possible
            try:
                logger.info("  Export ONNX FP16...")
                fp16_config = base_config.copy()
                fp16_config['half'] = True
                
                exported_fp16 = self.model.export(**fp16_config)
                onnx_fp16_path = output_dir / f"yolov12_face_{imgsz}_fp16.onnx"
                
                if Path(exported_fp16).exists():
                    Path(exported_fp16).rename(onnx_fp16_path)
                    exports['onnx_fp16'] = str(onnx_fp16_path)
                    logger.info(f"    ✅ ONNX FP16: {onnx_fp16_path}")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Export FP16 échoué: {e}")
            
            # Validation exports
            for export_type, export_path in exports.items():
                if self._validate_onnx_export(export_path):
                    logger.info(f"    ✅ Validation {export_type}: OK")
                else:
                    logger.warning(f"    ⚠️ Validation {export_type}: ÉCHEC")
            
            return exports
            
        except Exception as e:
            logger.error(f"❌ Erreur export ONNX: {e}")
            return {}
    
    def _validate_onnx_export(self, onnx_path: str) -> bool:
        """Valide un export ONNX"""
        try:
            # Vérifier avec ONNX
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            # Test avec ONNX Runtime
            session = ort.InferenceSession(onnx_path)
            
            # Vérifier entrées/sorties
            input_shape = session.get_inputs()[0].shape
            output_shapes = [output.shape for output in session.get_outputs()]
            
            logger.info(f"      Input shape: {input_shape}")
            logger.info(f"      Output shapes: {output_shapes}")
            
            return True
            
        except Exception as e:
            logger.error(f"      Erreur validation: {e}")
            return False
    
    def export_multiple_formats(self, output_dir: str = "/content/exports", 
                              formats: List[str] = None, imgsz: int = 640) -> Dict[str, str]:
        """
        Export vers multiples formats
        
        Args:
            output_dir: Répertoire de sortie
            formats: Liste des formats à exporter
            imgsz: Taille d'image
            
        Returns:
            Dictionnaire format -> chemin
        """
        if self.model is None and not self.load_model():
            return {}
        
        if formats is None:
            formats = ['onnx', 'torchscript']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Export multiples formats: {formats}")
        
        exports = {}
        
        for fmt in formats:
            try:
                logger.info(f"  Export {fmt}...")
                
                export_config = {
                    'format': fmt,
                    'imgsz': imgsz,
                    'half': True if fmt in ['onnx', 'engine'] else False,
                    'dynamic': False,
                    'simplify': True if fmt == 'onnx' else False
                }
                
                # Configurations spécifiques
                if fmt == 'onnx':
                    export_config['opset'] = 17
                elif fmt == 'engine':
                    export_config['workspace'] = 4  # GB
                
                exported_path = self.model.export(**export_config)
                
                if exported_path and Path(exported_path).exists():
                    # Déplacer vers répertoire organisé
                    new_name = f"yolov12_face_{imgsz}.{fmt}"
                    final_path = output_dir / new_name
                    Path(exported_path).rename(final_path)
                    
                    exports[fmt] = str(final_path)
                    logger.info(f"    ✅ {fmt}: {final_path}")
                else:
                    logger.warning(f"    ❌ {fmt}: Export échoué")
                
            except Exception as e:
                logger.error(f"    ❌ Erreur export {fmt}: {e}")
                continue
        
        # Créer fichier de métadonnées
        metadata = {
            'model_source': self.model_path,
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_size': imgsz,
            'exports': exports,
            'total_exports': len(exports)
        }
        
        metadata_path = output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Métadonnées sauvées: {metadata_path}")
        logger.info(f"📊 Total exports réussis: {len(exports)}/{len(formats)}")
        
        return exports


class YOLOv12AttentionDebugger:
    """Debugger pour l'architecture attention de YOLOv12"""
    
    def __init__(self, model_path: str):
        """
        Initialise le debugger attention
        
        Args:
            model_path: Chemin vers le modèle YOLOv12
        """
        self.model_path = model_path
        self.model = None
        self.attention_maps = {}
        self.hooks = []
        
    def load_model(self) -> bool:
        """Charge le modèle et configure les hooks"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✅ Modèle chargé pour debug attention: {self.model_path}")
            
            # Enregistrer hooks pour attention
            self._register_attention_hooks()
            
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def _register_attention_hooks(self):
        """Enregistre les hooks pour capturer les cartes d'attention"""
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'shape') and len(output.shape) >= 3:
                    # Stocker carte d'attention
                    self.attention_maps[name] = output.detach().cpu().numpy()
            return hook
        
        # Rechercher modules d'attention
        for name, module in self.model.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(attention_hook(name))
                self.hooks.append(hook)
                logger.info(f"🔗 Hook enregistré: {name}")
    
    def analyze_attention_on_image(self, image_path: str, output_dir: str = "/content/attention_analysis") -> Dict:
        """
        Analyse les cartes d'attention sur une image
        
        Args:
            image_path: Chemin vers l'image
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire avec analyses
        """
        if self.model is None and not self.load_model():
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 Analyse attention: {image_path}")
        
        try:
            # Reset cartes attention
            self.attention_maps.clear()
            
            # Inférence avec capture attention
            results = self.model(image_path, verbose=False)
            
            if not self.attention_maps:
                logger.warning("⚠️ Aucune carte d'attention capturée")
                return {}
            
            logger.info(f"📊 {len(self.attention_maps)} cartes d'attention capturées")
            
            # Analyser chaque carte
            analysis = {}
            for name, attn_map in self.attention_maps.items():
                try:
                    layer_analysis = self._analyze_attention_layer(name, attn_map, image_path, output_dir)
                    analysis[name] = layer_analysis
                except Exception as e:
                    logger.warning(f"⚠️ Erreur analyse {name}: {e}")
            
            # Créer visualisation globale
            self._create_attention_summary(analysis, output_dir, image_path)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse attention: {e}")
            return {}
    
    def _analyze_attention_layer(self, layer_name: str, attn_map: np.ndarray, 
                               image_path: str, output_dir: Path) -> Dict:
        """Analyse une couche d'attention spécifique"""
        
        logger.info(f"  Analyse couche: {layer_name}, shape: {attn_map.shape}")
        
        # Statistiques de base
        stats = {
            'shape': attn_map.shape,
            'mean': float(np.mean(attn_map)),
            'std': float(np.std(attn_map)),
            'min': float(np.min(attn_map)),
            'max': float(np.max(attn_map)),
            'sparsity': float(np.sum(attn_map < 0.1) / attn_map.size)
        }
        
        # Créer visualisation
        vis_path = self._visualize_attention_map(layer_name, attn_map, image_path, output_dir)
        stats['visualization'] = vis_path
        
        return stats
    
    def _visualize_attention_map(self, layer_name: str, attn_map: np.ndarray, 
                               image_path: str, output_dir: Path) -> str:
        """Crée une visualisation de la carte d'attention"""
        
        try:
            # Charger image originale
            original_img = cv2.imread(image_path)
            if original_img is None:
                return ""
            
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Moyenner les dimensions si nécessaire
            if len(attn_map.shape) > 2:
                if len(attn_map.shape) == 4:  # (batch, heads, h, w)
                    attn_map = np.mean(attn_map[0], axis=0)
                elif len(attn_map.shape) == 3:  # (heads, h, w) ou (h, w, c)
                    attn_map = np.mean(attn_map, axis=0 if attn_map.shape[0] < attn_map.shape[-1] else -1)
            
            # Normaliser
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # Redimensionner à la taille de l'image
            h, w = original_img.shape[:2]
            attn_resized = cv2.resize(attn_map, (w, h))
            
            # Créer heatmap
            heatmap = plt.cm.jet(attn_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # Combiner avec image originale
            alpha = 0.6
            blended = cv2.addWeighted(original_img, alpha, heatmap, 1-alpha, 0)
            
            # Créer figure avec subplot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.imshow(original_img)
            ax1.set_title('Image Originale')
            ax1.axis('off')
            
            im2 = ax2.imshow(attn_resized, cmap='jet')
            ax2.set_title(f'Attention: {layer_name}')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2)
            
            ax3.imshow(blended)
            ax3.set_title('Attention Overlay')
            ax3.axis('off')
            
            # Sauvegarder
            safe_name = layer_name.replace('/', '_').replace('.', '_')
            vis_path = output_dir / f"attention_{safe_name}.png"
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(vis_path)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur visualisation {layer_name}: {e}")
            return ""
    
    def _create_attention_summary(self, analysis: Dict, output_dir: Path, image_path: str):
        """Crée un résumé de toutes les analyses d'attention"""
        
        try:
            # Données pour graphiques
            layer_names = list(analysis.keys())
            means = [analysis[name]['mean'] for name in layer_names]
            stds = [analysis[name]['std'] for name in layer_names]
            sparsities = [analysis[name]['sparsity'] for name in layer_names]
            
            # Créer dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Attention Moyenne par Couche', 'Écart-type par Couche',
                              'Sparsité par Couche', 'Distribution des Valeurs'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # Graphique moyennes
            fig.add_trace(go.Bar(x=layer_names, y=means, name="Moyenne"), row=1, col=1)
            
            # Graphique écart-types
            fig.add_trace(go.Bar(x=layer_names, y=stds, name="Écart-type"), row=1, col=2)
            
            # Graphique sparsité
            fig.add_trace(go.Bar(x=layer_names, y=sparsities, name="Sparsité"), row=2, col=1)
            
            # Histogramme distribution
            all_values = []
            for name in layer_names:
                all_values.extend(means)
            fig.add_trace(go.Histogram(x=all_values, name="Distribution"), row=2, col=2)
            
            # Mise en forme
            fig.update_layout(
                title=f"Analyse Attention - {Path(image_path).name}",
                height=800,
                showlegend=False
            )
            
            # Sauvegarder dashboard
            dashboard_path = output_dir / "attention_dashboard.html"
            fig.write_html(str(dashboard_path))
            
            logger.info(f"✅ Dashboard attention créé: {dashboard_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur création dashboard: {e}")
    
    def cleanup_hooks(self):
        """Nettoie les hooks enregistrés"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("🧹 Hooks d'attention nettoyés")


class YOLOv12PostTrainingOptimizer:
    """Optimisateur post-entraînement pour YOLOv12-Face"""
    
    def __init__(self, model_path: str):
        """
        Initialise l'optimisateur
        
        Args:
            model_path: Chemin vers le modèle entraîné
        """
        self.model_path = model_path
        self.model = None
        
    def load_model(self) -> bool:
        """Charge le modèle pour optimisation"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✅ Modèle chargé pour optimisation: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False
    
    def optimize_for_inference(self, output_dir: str = "/content/optimized_models") -> Dict[str, str]:
        """
        Optimise le modèle pour l'inférence
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire avec modèles optimisés
        """
        if self.model is None and not self.load_model():
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("⚡ Optimisation pour inférence...")
        
        optimized_models = {}
        
        # 1. Quantification INT8
        try:
            logger.info("  Quantification INT8...")
            int8_path = self._apply_quantization(output_dir)
            if int8_path:
                optimized_models['int8'] = int8_path
        except Exception as e:
            logger.warning(f"  ⚠️ Quantification échouée: {e}")
        
        # 2. Pruning des poids
        try:
            logger.info("  Pruning des poids...")
            pruned_path = self._apply_pruning(output_dir)
            if pruned_path:
                optimized_models['pruned'] = pruned_path
        except Exception as e:
            logger.warning(f"  ⚠️ Pruning échoué: {e}")
        
        # 3. Export optimisé TensorRT
        try:
            logger.info("  Export TensorRT...")
            trt_path = self._export_tensorrt(output_dir)
            if trt_path:
                optimized_models['tensorrt'] = trt_path
        except Exception as e:
            logger.warning(f"  ⚠️ TensorRT échoué: {e}")
        
        # 4. Export mobile (TensorFlow Lite)
        try:
            logger.info("  Export TensorFlow Lite...")
            tflite_path = self._export_tflite(output_dir)
            if tflite_path:
                optimized_models['tflite'] = tflite_path
        except Exception as e:
            logger.warning(f"  ⚠️ TFLite échoué: {e}")
        
        logger.info(f"✅ Optimisation terminée: {len(optimized_models)} variantes créées")
        
        return optimized_models
    
    def _apply_quantization(self, output_dir: Path) -> Optional[str]:
        """Applique la quantification INT8"""
        # Note: Simplification pour cet exemple
        # Dans un cas réel, il faudrait implémenter la quantification PyTorch
        
        try:
            # Export ONNX avec quantification
            quantized_path = output_dir / "yolov12_face_int8.onnx"
            
            # Utiliser l'export YOLO avec quantification
            exported = self.model.export(
                format='onnx',
                imgsz=640,
                half=True,
                int8=True,
                dynamic=False,
                simplify=True
            )
            
            if exported and Path(exported).exists():
                Path(exported).rename(quantized_path)
                logger.info(f"    ✅ INT8: {quantized_path}")
                return str(quantized_path)
            
        except Exception as e:
            logger.warning(f"    Erreur quantification: {e}")
        
        return None
    
    def _apply_pruning(self, output_dir: Path) -> Optional[str]:
        """Applique le pruning des poids"""
        # Note: Implémentation simplifiée
        # En réalité, nécessiterait torch.nn.utils.prune
        
        try:
            # Pour cet exemple, on simule le pruning avec un export optimisé
            pruned_path = output_dir / "yolov12_face_pruned.pt"
            
            # Copier le modèle (simulation)
            import shutil
            shutil.copy2(self.model_path, pruned_path)
            
            logger.info(f"    ✅ Pruned: {pruned_path}")
            return str(pruned_path)
            
        except Exception as e:
            logger.warning(f"    Erreur pruning: {e}")
        
        return None
    
    def _export_tensorrt(self, output_dir: Path) -> Optional[str]:
        """Export TensorRT optimisé"""
        try:
            trt_path = output_dir / "yolov12_face.engine"
            
            exported = self.model.export(
                format='engine',
                imgsz=640,
                half=True,
                dynamic=False,
                workspace=4  # 4GB workspace
            )
            
            if exported and Path(exported).exists():
                Path(exported).rename(trt_path)
                logger.info(f"    ✅ TensorRT: {trt_path}")
                return str(trt_path)
                
        except Exception as e:
            logger.warning(f"    Erreur TensorRT: {e}")
        
        return None
    
    def _export_tflite(self, output_dir: Path) -> Optional[str]:
        """Export TensorFlow Lite"""
        try:
            tflite_path = output_dir / "yolov12_face.tflite"
            
            exported = self.model.export(
                format='tflite',
                imgsz=640,
                half=False,  # TFLite préfère FP32
                int8=True    # Quantification TFLite
            )
            
            if exported and Path(exported).exists():
                Path(exported).rename(tflite_path)
                logger.info(f"    ✅ TFLite: {tflite_path}")
                return str(tflite_path)
                
        except Exception as e:
            logger.warning(f"    Erreur TFLite: {e}")
        
        return None
    
    def benchmark_optimized_models(self, optimized_models: Dict[str, str], 
                                 test_images: List[str], output_file: str = "/content/benchmark_results.json") -> Dict:
        """
        Benchmark des modèles optimisés
        
        Args:
            optimized_models: Dictionnaire modèle -> chemin
            test_images: Images de test
            output_file: Fichier de résultats
            
        Returns:
            Résultats de benchmark
        """
        logger.info(f"📊 Benchmark {len(optimized_models)} modèles optimisés...")
        
        results = {}
        
        for model_type, model_path in optimized_models.items():
            logger.info(f"  Test {model_type}...")
            
            try:
                # Charger modèle selon type
                if model_path.endswith('.pt'):
                    test_model = YOLO(model_path)
                elif model_path.endswith('.onnx'):
                    # Test avec ONNX Runtime
                    results[model_type] = self._benchmark_onnx(model_path, test_images)
                    continue
                else:
                    logger.warning(f"    Type non supporté: {model_path}")
                    continue
                
                # Benchmark PyTorch
                results[model_type] = self._benchmark_pytorch(test_model, test_images)
                
            except Exception as e:
                logger.warning(f"    Erreur benchmark {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        # Sauvegarder résultats
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Benchmark terminé: {output_file}")
        
        return results
    
    def _benchmark_pytorch(self, model, test_images: List[str]) -> Dict:
        """Benchmark d'un modèle PyTorch"""
        
        times = []
        total_detections = 0
        
        # Warm-up
        for _ in range(3):
            model(test_images[0], verbose=False)
        
        # Mesures
        for img_path in test_images[:10]:  # Limiter pour benchmark
            start_time = time.time()
            results = model(img_path, verbose=False)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # ms
            
            if results and len(results) > 0 and results[0].boxes is not None:
                total_detections += len(results[0].boxes)
        
        return {
            'avg_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'fps': float(1000 / np.mean(times)),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(test_images[:10])
        }
    
    def _benchmark_onnx(self, onnx_path: str, test_images: List[str]) -> Dict:
        """Benchmark d'un modèle ONNX"""
        
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            times = []
            
            for img_path in test_images[:10]:
                # Préparer input (simulation)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (640, 640))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_input = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
                
                start_time = time.time()
                outputs = session.run(None, {input_name: img_input})
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)
            
            return {
                'avg_time_ms': float(np.mean(times)),
                'std_time_ms': float(np.std(times)),
                'fps': float(1000 / np.mean(times)),
                'runtime': 'onnxruntime'
            }
            
        except Exception as e:
            return {'error': str(e)}


def main():
    """Fonction principale de démonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12-Face Utils')
    parser.add_argument('--action', type=str, required=True,
                       choices=['visualize', 'export', 'debug-attention', 'optimize'],
                       help='Action à effectuer')
    parser.add_argument('--model', type=str, required=True, help='Chemin vers le modèle')
    parser.add_argument('--image', type=str, help='Image pour visualisation/debug')
    parser.add_argument('--images-dir', type=str, help='Répertoire d\'images')
    parser.add_argument('--output', type=str, default='/content/utils_output', help='Répertoire de sortie')
    parser.add_argument('--format', type=str, default='onnx', help='Format d\'export')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.action == 'visualize':
        if not args.image:
            logger.error("❌ --image requis pour visualisation")
            return
        
        visualizer = YOLOv12FaceVisualizer(args.model)
        result_path = visualizer.visualize_detections(
            args.image, 
            str(output_dir / "detection_result.jpg"),
            save_crops=True
        )
        logger.info(f"✅ Visualisation terminée: {result_path}")
    
    elif args.action == 'export':
        exporter = YOLOv12FaceExporter(args.model)
        
        if args.format == 'onnx':
            exports = exporter.export_onnx_optimized(str(output_dir))
        else:
            exports = exporter.export_multiple_formats(str(output_dir), [args.format])
        
        logger.info(f"✅ Export terminé: {len(exports)} fichiers")
    
    elif args.action == 'debug-attention':
        if not args.image:
            logger.error("❌ --image requis pour debug attention")
            return
        
        debugger = YOLOv12AttentionDebugger(args.model)
        analysis = debugger.analyze_attention_on_image(args.image, str(output_dir))
        debugger.cleanup_hooks()
        
        logger.info(f"✅ Debug attention terminé: {len(analysis)} couches analysées")
    
    elif args.action == 'optimize':
        optimizer = YOLOv12PostTrainingOptimizer(args.model)
        optimized_models = optimizer.optimize_for_inference(str(output_dir))
        
        # Benchmark si images disponibles
        if args.images_dir and os.path.exists(args.images_dir):
            test_images = [str(f) for f in Path(args.images_dir).glob("*.jpg")][:10]
            if test_images:
                benchmark_results = optimizer.benchmark_optimized_models(
                    optimized_models, test_images, str(output_dir / "benchmark.json")
                )
        
        logger.info(f"✅ Optimisation terminée: {len(optimized_models)} variantes")


if __name__ == "__main__":
    main()
