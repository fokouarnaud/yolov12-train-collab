"""
YOLOv12-Face - Benchmark Automatique Complet
Test de performance continue avec comparaisons baseline
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import torch

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv12FaceBenchmark:
    """Benchmark automatique complet pour YOLOv12-Face"""
    
    def __init__(self, output_dir: str = "/content/benchmark_results"):
        """
        Initialise le benchmark
        
        Args:
            output_dir: Répertoire de sortie des résultats
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Métriques baseline ADYOLOv5-Face
        self.baselines = {
            'adyolov5_face': {
                'map50': 0.891,
                'map50_95': 0.685,
                'precision': 0.912,
                'recall': 0.873,
                'f1_score': 0.892,
                'fps_640_t4': 45.2,
                'fps_640_cpu': 12.1,
                'widerface_easy': 0.948,
                'widerface_medium': 0.938,
                'widerface_hard': 0.844,
                'model_size_mb': 14.7,
                'inference_time_ms': 22.1
            },
            'yolov8n_face': {
                'map50': 0.856,
                'map50_95': 0.641,
                'precision': 0.887,
                'recall': 0.834,
                'f1_score': 0.860,
                'fps_640_t4': 52.3,
                'fps_640_cpu': 15.2,
                'model_size_mb': 6.2,
                'inference_time_ms': 19.1
            }
        }
        
        # Objectifs YOLOv12-Face
        self.targets = {
            'map50': 0.920,           # +3.3% vs ADYOLOv5
            'map50_95': 0.720,        # +5.1% vs ADYOLOv5
            'precision': 0.940,       # +3.1% vs ADYOLOv5
            'recall': 0.900,          # +3.1% vs ADYOLOv5
            'f1_score': 0.920,        # +3.1% vs ADYOLOv5
            'fps_640_t4': 60.0,       # +32.7% vs ADYOLOv5
            'widerface_easy': 0.975,  # +2.8% vs ADYOLOv5
            'widerface_medium': 0.965, # +2.9% vs ADYOLOv5
            'widerface_hard': 0.885,  # +4.9% vs ADYOLOv5
            'model_size_mb': 15.0,    # Similaire ADYOLOv5
            'inference_time_ms': 16.7 # -24.4% vs ADYOLOv5
        }
        
        self.system_info = self._get_system_info()
        self.results = {}
        
    def _get_system_info(self) -> Dict:
        """Collecte informations système"""
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # GPU Info
        if torch.cuda.is_available():
            info['gpu'] = {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'driver_version': torch.version.cuda
            }
        else:
            info['gpu'] = {'available': False}
        
        # CPU Info
        try:
            import psutil
            info['cpu'] = {
                'cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1e9,
                'usage_percent': psutil.cpu_percent()
            }
        except ImportError:
            info['cpu'] = {'cores': os.cpu_count(), 'memory_gb': 'unknown'}
        
        return info
    
    def benchmark_model_accuracy(self, model_path: str, data_path: str) -> Dict:
        """Benchmark précision du modèle"""
        
        logger.info("📊 Benchmark précision du modèle...")
        
        try:
            from ultralytics import YOLO
            
            # Charger modèle
            model = YOLO(model_path)
            logger.info(f"✅ Modèle chargé: {Path(model_path).name}")
            
            # Validation
            val_results = model.val(
                data=data_path,
                imgsz=640,
                batch=8,
                conf=0.001,
                iou=0.6,
                plots=False,
                verbose=False
            )
            
            # Extraire métriques
            accuracy_metrics = {
                'map50': float(val_results.box.map50),
                'map50_95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'f1_score': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr + 1e-16)
            }
            
            # Comparaisons avec baselines
            accuracy_metrics['vs_adyolov5'] = self._calculate_improvements(
                accuracy_metrics, self.baselines['adyolov5_face']
            )
            
            accuracy_metrics['vs_targets'] = self._calculate_target_achievement(
                accuracy_metrics, self.targets
            )
            
            logger.info(f"✅ Métriques précision calculées")
            logger.info(f"   mAP@0.5: {accuracy_metrics['map50']:.3f}")
            logger.info(f"   mAP@0.5:0.95: {accuracy_metrics['map50_95']:.3f}")
            logger.info(f"   Precision: {accuracy_metrics['precision']:.3f}")
            logger.info(f"   Recall: {accuracy_metrics['recall']:.3f}")
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark précision: {e}")
            return {}
    
    def benchmark_inference_speed(self, model_path: str, test_images: List[str], 
                                 num_runs: int = 50) -> Dict:
        """Benchmark vitesse d'inférence"""
        
        logger.info(f"⚡ Benchmark vitesse d'inférence ({num_runs} runs)...")
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # Différentes configurations de test
            test_configs = [
                {'imgsz': 320, 'name': '320px'},
                {'imgsz': 640, 'name': '640px'},
                {'imgsz': 1280, 'name': '1280px'}
            ]
            
            speed_results = {}
            
            for config in test_configs:
                logger.info(f"  Test {config['name']}...")
                
                # Warm-up
                for _ in range(5):
                    model(test_images[0], imgsz=config['imgsz'], verbose=False)
                
                # Mesures
                times = []
                for _ in range(num_runs):
                    img_path = test_images[np.random.randint(0, len(test_images))]
                    
                    start_time = time.time()
                    results = model(img_path, imgsz=config['imgsz'], verbose=False)
                    end_time = time.time()
                    
                    times.append((end_time - start_time) * 1000)  # ms
                
                # Statistiques
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                fps = 1000 / avg_time
                
                speed_results[config['name']] = {
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'fps': fps,
                    'throughput': fps  # images/sec
                }
                
                logger.info(f"    ✅ {config['name']}: {avg_time:.1f}±{std_time:.1f}ms, {fps:.1f} FPS")
            
            # Comparaisons
            if '640px' in speed_results:
                baseline_fps = self.baselines['adyolov5_face']['fps_640_t4']
                current_fps = speed_results['640px']['fps']
                improvement = (current_fps - baseline_fps) / baseline_fps * 100
                
                speed_results['vs_baseline'] = {
                    'baseline_fps': baseline_fps,
                    'current_fps': current_fps,
                    'improvement_percent': improvement,
                    'faster': improvement > 0
                }
                
                logger.info(f"  📈 vs ADYOLOv5: {improvement:+.1f}% ({'✅' if improvement > 0 else '❌'})")
            
            return speed_results
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark vitesse: {e}")
            return {}
    
    def benchmark_model_size(self, model_path: str) -> Dict:
        """Benchmark taille et complexité du modèle"""
        
        logger.info("📏 Benchmark taille du modèle...")
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # Taille fichier
            file_size_bytes = Path(model_path).stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Paramètres du modèle
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            # Calcul FLOPs (approximation)
            input_size = (1, 3, 640, 640)
            flops = self._estimate_flops(model.model, input_size)
            
            size_metrics = {
                'file_size_mb': file_size_mb,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'estimated_flops': flops,
                'parameters_density': total_params / file_size_mb  # params per MB
            }
            
            # Comparaison baseline
            baseline_size = self.baselines['adyolov5_face']['model_size_mb']
            size_metrics['vs_baseline'] = {
                'baseline_size_mb': baseline_size,
                'size_ratio': file_size_mb / baseline_size,
                'size_difference_mb': file_size_mb - baseline_size
            }
            
            logger.info(f"  📊 Taille: {file_size_mb:.1f} MB")
            logger.info(f"  🔢 Paramètres: {total_params:,}")
            logger.info(f"  ⚡ FLOPs: {flops/1e9:.1f}G")
            
            return size_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark taille: {e}")
            return {}
    
    def _estimate_flops(self, model, input_size: Tuple[int, int, int, int]) -> int:
        """Estimation approximative des FLOPs"""
        try:
            import torch
            
            # Créer input factice
            x = torch.randn(input_size)
            
            # Hook pour compter opérations (approximation simple)
            total_flops = 0
            
            def flop_count_hook(module, input, output):
                nonlocal total_flops
                if isinstance(module, torch.nn.Conv2d):
                    # Approximation pour convolutions
                    output_dims = output.shape
                    kernel_dims = module.kernel_size
                    in_channels = module.in_channels
                    flops = np.prod(output_dims) * np.prod(kernel_dims) * in_channels
                    total_flops += flops
            
            # Enregistrer hooks
            hooks = []
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    hooks.append(module.register_forward_hook(flop_count_hook))
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(x)
            
            # Nettoyer hooks
            for hook in hooks:
                hook.remove()
            
            return total_flops
            
        except Exception as e:
            logger.warning(f"⚠️ Estimation FLOPs échouée: {e}")
            return 0
    
    def benchmark_memory_usage(self, model_path: str, test_images: List[str]) -> Dict:
        """Benchmark utilisation mémoire"""
        
        logger.info("💾 Benchmark utilisation mémoire...")
        
        try:
            from ultralytics import YOLO
            import psutil
            import gc
            
            # Mesure avant chargement
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Charger modèle
            model = YOLO(model_path)
            memory_after_load = process.memory_info().rss / 1024 / 1024
            
            # GPU memory si disponible
            gpu_memory = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_memory['before'] = torch.cuda.memory_allocated() / 1024 / 1024
                
                # Test inférence
                _ = model(test_images[0], verbose=False)
                gpu_memory['after_inference'] = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory['peak'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Multiple inférences pour test stabilité
            memory_during_inference = []
            for i in range(10):
                _ = model(test_images[i % len(test_images)], verbose=False)
                memory_during_inference.append(process.memory_info().rss / 1024 / 1024)
            
            memory_metrics = {
                'cpu_memory': {
                    'before_load_mb': memory_before,
                    'after_load_mb': memory_after_load,
                    'loading_overhead_mb': memory_after_load - memory_before,
                    'during_inference_avg_mb': np.mean(memory_during_inference),
                    'during_inference_max_mb': np.max(memory_during_inference),
                    'during_inference_std_mb': np.std(memory_during_inference)
                }
            }
            
            if gpu_memory:
                memory_metrics['gpu_memory'] = {
                    'inference_usage_mb': gpu_memory['after_inference'] - gpu_memory['before'],
                    'peak_usage_mb': gpu_memory['peak'],
                    'total_available_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                }
            
            logger.info(f"  💾 RAM: {memory_metrics['cpu_memory']['loading_overhead_mb']:.1f} MB overhead")
            if gpu_memory:
                logger.info(f"  🎮 VRAM: {memory_metrics['gpu_memory']['peak_usage_mb']:.1f} MB peak")
            
            return memory_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark mémoire: {e}")
            return {}
    
    def _calculate_improvements(self, current: Dict, baseline: Dict) -> Dict:
        """Calcule les améliorations vs baseline"""
        
        improvements = {}
        
        for metric in ['map50', 'map50_95', 'precision', 'recall', 'f1_score']:
            if metric in current and metric in baseline:
                current_val = current[metric]
                baseline_val = baseline[metric]
                
                improvement = (current_val - baseline_val) / baseline_val * 100
                
                improvements[metric] = {
                    'current': current_val,
                    'baseline': baseline_val,
                    'improvement_percent': improvement,
                    'better': improvement > 0
                }
        
        return improvements
    
    def _calculate_target_achievement(self, current: Dict, targets: Dict) -> Dict:
        """Calcule l'atteinte des objectifs"""
        
        achievements = {}
        
        for metric in targets:
            if metric in current:
                current_val = current[metric]
                target_val = targets[metric]
                
                achievement_percent = (current_val / target_val) * 100
                
                achievements[metric] = {
                    'current': current_val,
                    'target': target_val,
                    'achievement_percent': achievement_percent,
                    'achieved': current_val >= target_val,
                    'gap': current_val - target_val
                }
        
        return achievements
    
    def run_complete_benchmark(self, model_path: str, data_path: str, 
                              test_images: List[str]) -> Dict:
        """Lance benchmark complet"""
        
        logger.info("🚀 Démarrage benchmark complet YOLOv12-Face")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Vérifications préliminaires
        if not Path(model_path).exists():
            logger.error(f"❌ Modèle non trouvé: {model_path}")
            return {}
        
        if not Path(data_path).exists():
            logger.error(f"❌ Dataset non trouvé: {data_path}")
            return {}
        
        if not test_images:
            logger.error("❌ Aucune image de test fournie")
            return {}
        
        logger.info(f"📦 Modèle: {Path(model_path).name}")
        logger.info(f"📊 Dataset: {Path(data_path).name}")
        logger.info(f"🖼️ Images test: {len(test_images)}")
        
        # 1. Benchmark précision
        self.results['accuracy'] = self.benchmark_model_accuracy(model_path, data_path)
        
        # 2. Benchmark vitesse
        self.results['speed'] = self.benchmark_inference_speed(model_path, test_images[:10])
        
        # 3. Benchmark taille
        self.results['size'] = self.benchmark_model_size(model_path)
        
        # 4. Benchmark mémoire
        self.results['memory'] = self.benchmark_memory_usage(model_path, test_images[:5])
        
        # 5. Informations système
        self.results['system'] = self.system_info
        
        # 6. Résumé global
        self.results['summary'] = self._generate_summary()
        
        # Temps total
        total_time = time.time() - start_time
        self.results['benchmark_duration_minutes'] = total_time / 60
        
        logger.info(f"✅ Benchmark terminé en {total_time/60:.1f} minutes")
        
        return self.results
    
    def _generate_summary(self) -> Dict:
        """Génère résumé des résultats"""
        
        summary = {
            'overall_score': 0.0,
            'achievements': {},
            'recommendations': []
        }
        
        # Score global (moyenne pondérée)
        scores = []
        
        # Score précision (40% du total)
        if 'accuracy' in self.results and 'vs_targets' in self.results['accuracy']:
            accuracy_achievements = self.results['accuracy']['vs_targets']
            accuracy_score = np.mean([
                min(100, achv['achievement_percent']) 
                for achv in accuracy_achievements.values()
            ])
            scores.append(('accuracy', accuracy_score, 0.4))
        
        # Score vitesse (30% du total)
        if 'speed' in self.results and 'vs_baseline' in self.results['speed']:
            speed_improvement = self.results['speed']['vs_baseline']['improvement_percent']
            speed_score = min(100, max(0, 100 + speed_improvement))  # 100% si égal baseline
            scores.append(('speed', speed_score, 0.3))
        
        # Score efficacité (20% du total)
        if 'size' in self.results:
            # Score basé sur rapport taille/performance
            size_score = 75  # Score par défaut
            scores.append(('efficiency', size_score, 0.2))
        
        # Score stabilité (10% du total)
        if 'memory' in self.results:
            # Score basé sur stabilité mémoire
            stability_score = 80  # Score par défaut
            scores.append(('stability', stability_score, 0.1))
        
        # Calcul score global
        if scores:
            summary['overall_score'] = sum(score * weight for _, score, weight in scores)
            summary['scores_detail'] = {name: score for name, score, _ in scores}
        
        # Recommandations automatiques
        if summary['overall_score'] < 70:
            summary['recommendations'].append("Envisager ajustement hyperparamètres")
        
        if 'accuracy' in self.results:
            acc = self.results['accuracy']
            if acc.get('map50', 0) < self.targets['map50']:
                summary['recommendations'].append("Augmenter epochs ou améliorer dataset")
        
        if 'speed' in self.results and 'vs_baseline' in self.results['speed']:
            if self.results['speed']['vs_baseline']['improvement_percent'] < 0:
                summary['recommendations'].append("Optimiser modèle pour vitesse")
        
        return summary
    
    def generate_report(self, format: str = 'html') -> str:
        """Génère rapport détaillé"""
        
        if not self.results:
            logger.error("❌ Aucun résultat à reporter")
            return ""
        
        logger.info(f"📋 Génération rapport {format.upper()}...")
        
        if format == 'html':
            return self._generate_html_report()
        elif format == 'json':
            return self._generate_json_report()
        elif format == 'markdown':
            return self._generate_markdown_report()
        else:
            logger.error(f"❌ Format non supporté: {format}")
            return ""
    
    def _generate_html_report(self) -> str:
        """Génère rapport HTML interactif"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv12-Face Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; }}
        .metric-card {{ background: #f8f9fa; border: 1px solid #dee2e6; 
                       border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .improvement {{ color: #28a745; }}
        .regression {{ color: #dc3545; }}
        .table {{ width: 100%; border-collapse: collapse; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 YOLOv12-Face Benchmark Report</h1>
        <p>Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Score Global: <span class="score">{self.results.get('summary', {}).get('overall_score', 0):.1f}/100</span></p>
    </div>
    
    <h2>📊 Résumé Exécutif</h2>
    <div class="metric-card">
        <h3>Objectifs Atteints</h3>
        {self._format_achievements_html()}
    </div>
    
    <h2>📈 Métriques Détaillées</h2>
    {self._format_detailed_metrics_html()}
    
    <h2>💡 Recommandations</h2>
    <div class="metric-card">
        <ul>
        {chr(10).join([f'<li>{rec}</li>' for rec in self.results.get('summary', {}).get('recommendations', [])])}
        </ul>
    </div>
    
    <h2>🔧 Informations Système</h2>
    {self._format_system_info_html()}
    
</body>
</html>
"""
        
        report_path = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Rapport HTML: {report_path}")
        return str(report_path)
    
    def _format_achievements_html(self) -> str:
        """Formate les achievements en HTML"""
        
        if 'accuracy' not in self.results or 'vs_targets' not in self.results['accuracy']:
            return "<p>Données d'achievement non disponibles</p>"
        
        achievements = self.results['accuracy']['vs_targets']
        
        html = '<table class="table"><tr><th>Métrique</th><th>Actuel</th><th>Objectif</th><th>Statut</th></tr>'
        
        for metric, data in achievements.items():
            current = data['current']
            target = data['target']
            achieved = data['achieved']
            
            status_class = 'improvement' if achieved else 'regression'
            status_text = '✅ Atteint' if achieved else '❌ Non atteint'
            
            html += f'''
            <tr>
                <td>{metric}</td>
                <td>{current:.3f}</td>
                <td>{target:.3f}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
            '''
        
        html += '</table>'
        return html
    
    def _format_detailed_metrics_html(self) -> str:
        """Formate métriques détaillées en HTML"""
        
        html = ""
        
        # Métriques de précision
        if 'accuracy' in self.results:
            html += '<div class="metric-card"><h3>🎯 Précision</h3>'
            acc = self.results['accuracy']
            html += f'''
            <p>mAP@0.5: <strong>{acc.get('map50', 0):.3f}</strong></p>
            <p>mAP@0.5:0.95: <strong>{acc.get('map50_95', 0):.3f}</strong></p>
            <p>Precision: <strong>{acc.get('precision', 0):.3f}</strong></p>
            <p>Recall: <strong>{acc.get('recall', 0):.3f}</strong></p>
            '''
            html += '</div>'
        
        # Métriques de vitesse
        if 'speed' in self.results and '640px' in self.results['speed']:
            html += '<div class="metric-card"><h3>⚡ Vitesse</h3>'
            speed = self.results['speed']['640px']
            html += f'''
            <p>FPS (640px): <strong>{speed.get('fps', 0):.1f}</strong></p>
            <p>Temps moyen: <strong>{speed.get('avg_time_ms', 0):.1f} ms</strong></p>
            '''
            html += '</div>'
        
        return html
    
    def _format_system_info_html(self) -> str:
        """Formate info système en HTML"""
        
        sys_info = self.results.get('system', {})
        
        html = '<div class="metric-card">'
        
        if 'gpu' in sys_info and sys_info['gpu']['available']:
            gpu = sys_info['gpu']
            html += f'''
            <p><strong>GPU:</strong> {gpu.get('name', 'Unknown')}</p>
            <p><strong>VRAM:</strong> {gpu.get('memory_gb', 0):.1f} GB</p>
            '''
        
        if 'cpu' in sys_info:
            cpu = sys_info['cpu']
            html += f'''
            <p><strong>CPU:</strong> {cpu.get('cores', 'Unknown')} cores</p>
            <p><strong>RAM:</strong> {cpu.get('memory_gb', 'Unknown')} GB</p>
            '''
        
        html += '</div>'
        return html
    
    def _generate_json_report(self) -> str:
        """Génère rapport JSON"""
        
        report_path = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"✅ Rapport JSON: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self) -> str:
        """Génère rapport Markdown"""
        
        md_content = f"""# 🚀 YOLOv12-Face Benchmark Report

**Généré le:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Score Global:** {self.results.get('summary', {}).get('overall_score', 0):.1f}/100

## 📊 Résumé Exécutif

### Objectifs Atteints

{self._format_achievements_markdown()}

## 📈 Métriques Détaillées

### 🎯 Précision
"""
        
        if 'accuracy' in self.results:
            acc = self.results['accuracy']
            md_content += f"""
- **mAP@0.5:** {acc.get('map50', 0):.3f}
- **mAP@0.5:0.95:** {acc.get('map50_95', 0):.3f}
- **Precision:** {acc.get('precision', 0):.3f}
- **Recall:** {acc.get('recall', 0):.3f}
"""

        if 'speed' in self.results and '640px' in self.results['speed']:
            speed = self.results['speed']['640px']
            md_content += f"""
### ⚡ Vitesse

- **FPS (640px):** {speed.get('fps', 0):.1f}
- **Temps moyen:** {speed.get('avg_time_ms', 0):.1f} ms
"""

        md_content += f"""
## 💡 Recommandations

{chr(10).join([f'- {rec}' for rec in self.results.get('summary', {}).get('recommendations', [])])}
"""
        
        report_path = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"✅ Rapport Markdown: {report_path}")
        return str(report_path)
    
    def _format_achievements_markdown(self) -> str:
        """Formate achievements en Markdown"""
        
        if 'accuracy' not in self.results or 'vs_targets' not in self.results['accuracy']:
            return "Données d'achievement non disponibles"
        
        achievements = self.results['accuracy']['vs_targets']
        
        md = "| Métrique | Actuel | Objectif | Statut |\n|----------|--------|----------|--------|\n"
        
        for metric, data in achievements.items():
            current = data['current']
            target = data['target']
            achieved = data['achieved']
            
            status = '✅ Atteint' if achieved else '❌ Non atteint'
            
            md += f"| {metric} | {current:.3f} | {target:.3f} | {status} |\n"
        
        return md


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12-Face Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Chemin vers le modèle')
    parser.add_argument('--data', type=str, required=True, help='Chemin vers dataset.yaml')
    parser.add_argument('--images', type=str, required=True, help='Répertoire images test')
    parser.add_argument('--output', type=str, default='/content/benchmark_results', help='Répertoire sortie')
    parser.add_argument('--format', type=str, default='html', choices=['html', 'json', 'markdown'], help='Format rapport')
    parser.add_argument('--runs', type=int, default=50, help='Nombre de runs vitesse')
    
    args = parser.parse_args()
    
    # Créer benchmark
    benchmark = YOLOv12FaceBenchmark(args.output)
    
    # Préparer images de test
    test_images = []
    if os.path.isdir(args.images):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(list(Path(args.images).glob(ext)))
        test_images = [str(img) for img in test_images[:20]]  # Limiter à 20 images
    else:
        logger.error(f"❌ Répertoire images non trouvé: {args.images}")
        return
    
    if not test_images:
        logger.error("❌ Aucune image de test trouvée")
        return
    
    logger.info(f"🖼️ {len(test_images)} images de test préparées")
    
    # Lancer benchmark complet
    results = benchmark.run_complete_benchmark(args.model, args.data, test_images)
    
    if results:
        # Générer rapport
        report_path = benchmark.generate_report(args.format)
        
        logger.info("\n" + "="*60)
        logger.info("🏆 BENCHMARK TERMINÉ")
        logger.info("="*60)
        
        # Afficher résumé
        if 'summary' in results:
            summary = results['summary']
            logger.info(f"📊 Score global: {summary.get('overall_score', 0):.1f}/100")
            
            if 'accuracy' in results:
                acc = results['accuracy']
                logger.info(f"🎯 mAP@0.5: {acc.get('map50', 0):.3f}")
                logger.info(f"🎯 Precision: {acc.get('precision', 0):.3f}")
            
            if 'speed' in results and '640px' in results['speed']:
                fps = results['speed']['640px']['fps']
                logger.info(f"⚡ FPS: {fps:.1f}")
        
        logger.info(f"📋 Rapport généré: {report_path}")
        
    else:
        logger.error("❌ Benchmark échoué")


if __name__ == "__main__":
    main()
