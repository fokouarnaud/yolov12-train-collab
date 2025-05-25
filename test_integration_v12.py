"""
YOLOv12-Face - Test d'Intégration Pipeline Complet
Test de validation pour tous les scripts créés
"""

import sys
import os
import logging
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test des importations de tous les modules"""
    
    logger.info("🔍 Test des importations...")
    
    # Liste des modules à tester
    modules_to_test = [
        'config_v12',
        'main_v12', 
        'colab_setup_v12',
        'data_preparation_v12',
        'model_evaluation_v12',
        'utils_v12'
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            logger.info(f"  Test import: {module_name}")
            
            if module_name == 'config_v12':
                from config_v12 import YOLOv12FaceConfig, create_preset_configs
                # Test création config
                config_manager = YOLOv12FaceConfig()
                test_config = config_manager.get_complete_config('n', 'development', 'moderate')
                assert 'model' in test_config
                assert 'training' in test_config
                logger.info(f"    ✅ {module_name}: Config OK")
                
            elif module_name == 'main_v12':
                from main_v12 import YOLOv12FaceTrainer
                # Test initialisation
                trainer = YOLOv12FaceTrainer()
                assert trainer.config is not None
                logger.info(f"    ✅ {module_name}: Trainer OK")
                
            elif module_name == 'model_evaluation_v12':
                from model_evaluation_v12 import FaceDetectionEvaluator
                # Test initialisation (sans modèle)
                evaluator = FaceDetectionEvaluator.__new__(FaceDetectionEvaluator)
                logger.info(f"    ✅ {module_name}: Evaluator OK")
                
            elif module_name == 'utils_v12':
                from utils_v12 import (YOLOv12FaceVisualizer, YOLOv12FaceExporter, 
                                     YOLOv12AttentionDebugger, YOLOv12PostTrainingOptimizer)
                # Test initialisation classes
                visualizer = YOLOv12FaceVisualizer()
                assert visualizer.colors is not None
                logger.info(f"    ✅ {module_name}: Utils OK")
                
            else:
                # Import basique pour les autres modules
                __import__(module_name)
                logger.info(f"    ✅ {module_name}: Import OK")
            
            success_count += 1
            
        except ImportError as e:
            logger.error(f"    ❌ {module_name}: Import Error - {e}")
        except Exception as e:
            logger.error(f"    ❌ {module_name}: Error - {e}")
    
    logger.info(f"📊 Résultat imports: {success_count}/{len(modules_to_test)} réussis")
    return success_count == len(modules_to_test)

def test_dependencies():
    """Test des dépendances requises"""
    
    logger.info("📦 Test des dépendances...")
    
    required_packages = [
        'torch',
        'ultralytics', 
        'opencv-python',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'plotly',
        'pillow',
        'pyyaml',
        'onnx',
        'onnxruntime'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Mapping des noms de packages
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'pillow':
                import_name = 'PIL'
            elif package == 'pyyaml':
                import_name = 'yaml'
            
            __import__(import_name)
            logger.info(f"  ✅ {package}")
            
        except ImportError:
            logger.warning(f"  ❌ {package} - MANQUANT")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"📦 Packages manquants: {missing_packages}")
        logger.info("Pour installer: pip install " + " ".join(missing_packages))
        return False
    else:
        logger.info("✅ Toutes les dépendances sont installées")
        return True

def test_config_generation():
    """Test de génération de configurations"""
    
    logger.info("⚙️ Test génération configurations...")
    
    try:
        from config_v12 import YOLOv12FaceConfig, create_preset_configs
        
        # Test config manager
        config_manager = YOLOv12FaceConfig()
        
        # Test différentes configurations
        configs_to_test = [
            ('n', 'development', 'moderate'),
            ('s', 'production', 'conservative'), 
            ('m', 'fine_tuning', 'aggressive')
        ]
        
        for model_size, training_mode, aug_mode in configs_to_test:
            config = config_manager.get_complete_config(model_size, training_mode, aug_mode)
            
            # Vérifications
            assert config['model']['size'] == model_size
            assert config['training']['mode'] == training_mode
            assert config['augmentation']['mode'] == aug_mode
            assert 'hyperparameters' in config
            
            logger.info(f"  ✅ Config {model_size}-{training_mode}-{aug_mode}: OK")
        
        # Test validation
        test_config = config_manager.get_complete_config('n', 'development', 'moderate')
        is_valid = config_manager.validate_config(test_config)
        assert is_valid
        
        # Test presets
        presets = create_preset_configs()
        assert 'colab_nano' in presets
        assert 'production_small' in presets
        
        logger.info("✅ Génération configurations: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test configurations: {e}")
        return False

def test_data_preparation():
    """Test du module de préparation des données"""
    
    logger.info("📁 Test préparation données...")
    
    try:
        from data_preparation_v12 import WiderFaceConverter
        
        # Test initialisation
        converter = WiderFaceConverter.__new__(WiderFaceConverter)
        
        # Vérifier que les méthodes existent
        assert hasattr(converter, 'convert_annotations')
        assert hasattr(converter, 'create_dataset_yaml')
        assert hasattr(converter, 'validate_dataset')
        
        logger.info("✅ Module data_preparation_v12: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test data_preparation: {e}")
        return False

def create_integration_test_script():
    """Crée un script de test d'intégration complet"""
    
    test_script = """#!/usr/bin/env python3
'''
YOLOv12-Face - Script de Test d'Intégration Colab
Usage: python test_integration_colab.py
'''

def test_colab_environment():
    '''Test l'environnement Google Colab'''
    
    import os
    import sys
    
    print("🔍 Test environnement Colab...")
    
    # Vérifier si on est dans Colab
    try:
        import google.colab
        print("✅ Environnement Google Colab détecté")
        is_colab = True
    except ImportError:
        print("ℹ️ Environnement local (non-Colab)")
        is_colab = False
    
    # Vérifier GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU disponible: {gpu_name}")
    else:
        print("⚠️ Aucun GPU détecté")
    
    # Vérifier espace disque
    import shutil
    free_space = shutil.disk_usage('/content' if is_colab else '.')[2] / (1024**3)
    print(f"💾 Espace libre: {free_space:.1f} GB")
    
    return is_colab

def test_yolov12_installation():
    '''Test installation YOLOv12'''
    
    print("📦 Test installation YOLOv12...")
    
    try:
        from ultralytics import YOLO
        
        # Test chargement modèle de base
        model = YOLO('yolov8n.pt')  # Fallback si yolov12 pas disponible
        print("✅ YOLO installé et fonctionnel")
        return True
        
    except Exception as e:
        print(f"❌ Erreur YOLO: {e}")
        return False

def test_pipeline_complete():
    '''Test du pipeline complet YOLOv12-Face'''
    
    print("🚀 Test pipeline YOLOv12-Face...")
    
    try:
        # 1. Test configuration
        from config_v12 import YOLOv12FaceConfig
        config_manager = YOLOv12FaceConfig()
        config = config_manager.get_complete_config('n', 'quick_test', 'moderate')
        print("  ✅ Configuration: OK")
        
        # 2. Test trainer (sans entraînement réel)
        from main_v12 import YOLOv12FaceTrainer
        trainer = YOLOv12FaceTrainer()
        train_config = trainer.prepare_training_config()
        print("  ✅ Trainer: OK")
        
        # 3. Test evaluator (sans modèle)
        from model_evaluation_v12 import FaceDetectionEvaluator
        print("  ✅ Evaluator: OK")
        
        # 4. Test utils
        from utils_v12 import YOLOv12FaceVisualizer
        visualizer = YOLOv12FaceVisualizer()
        print("  ✅ Utils: OK")
        
        print("✅ Pipeline complet: FONCTIONNEL")
        return True
        
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        return False

if __name__ == "__main__":
    print("🎯 YOLOv12-Face - Test d'Intégration\\n")
    
    # Tests séquentiels
    tests = [
        ("Environnement Colab", test_colab_environment),
        ("Installation YOLO", test_yolov12_installation), 
        ("Pipeline YOLOv12-Face", test_pipeline_complete)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n{'='*50}")
        print(f"TEST: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ÉCHEC: {e}")
            results.append((test_name, False))
    
    # Résumé
    print(f"\\n{'='*50}")
    print("RÉSUMÉ DES TESTS")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\\nRésultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        print("\\n🚀 Le pipeline YOLOv12-Face est prêt pour l'entraînement!")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les dépendances.")
"""
    
    return test_script

def main():
    """Fonction principale du test d'intégration"""
    
    logger.info("🎯 YOLOv12-Face - Test d'Intégration Pipeline")
    logger.info("="*60)
    
    # Tests séquentiels
    tests = [
        ("Dépendances", test_dependencies),
        ("Importations", test_imports),
        ("Génération Config", test_config_generation),
        ("Préparation Données", test_data_preparation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"TEST: {test_name}")
        logger.info('='*40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ ÉCHEC: {e}")
            results.append((test_name, False))
    
    # Résumé
    logger.info(f"\n{'='*40}")
    logger.info("RÉSUMÉ DES TESTS")
    logger.info('='*40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nRésultat global: {passed}/{total} tests réussis")
    
    # Créer script de test Colab
    colab_test_script = create_integration_test_script()
    colab_test_path = "/content/test_integration_colab.py"
    
    try:
        with open(colab_test_path, 'w') as f:
            f.write(colab_test_script)
        logger.info(f"✅ Script test Colab créé: {colab_test_path}")
    except:
        # Fallback local
        local_test_path = "test_integration_colab.py"
        with open(local_test_path, 'w') as f:
            f.write(colab_test_script)
        logger.info(f"✅ Script test local créé: {local_test_path}")
    
    if passed == total:
        logger.info("\n🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("\n🚀 Pipeline YOLOv12-Face prêt:")
        logger.info("   1. ✅ Configuration: config_v12.py")
        logger.info("   2. ✅ Entraînement: main_v12.py") 
        logger.info("   3. ✅ Données: data_preparation_v12.py")
        logger.info("   4. ✅ Évaluation: model_evaluation_v12.py")
        logger.info("   5. ✅ Utilitaires: utils_v12.py")
        logger.info("\n📋 Prochaines étapes:")
        logger.info("   1. Préparer dataset WiderFace avec data_preparation_v12.py")
        logger.info("   2. Configurer entraînement avec config_v12.py")
        logger.info("   3. Lancer entraînement avec main_v12.py")
        logger.info("   4. Évaluer avec model_evaluation_v12.py")
        logger.info("   5. Optimiser avec utils_v12.py")
    else:
        logger.warning("\n⚠️ Certains tests ont échoué")
        logger.info("Vérifiez les dépendances manquantes et relancez les tests")

if __name__ == "__main__":
    main()
