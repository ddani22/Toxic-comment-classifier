import joblib
import os
import sys

def smoke_test_artifacts():
    """Smoke test que carga artifacts/ y ejecuta predicción de muestra"""
    print("🧪 SMOKE TEST: Artifact Loading")
    print("=" * 40)
    
    # Verificar artifacts según copilot-instructions.md
    expected_artifacts = {
        'artifacts/model.pkl': None,
        'artifacts/vectorizer.pkl': None,
        'artifacts/detailed_model.pkl': None,  # Nombre real en artifacts/
        'artifacts/vectorizer_detailed.pkl': None,
        'artifacts/toxicity_categories.pkl': None
    }
    
    # Verificar artifacts según copilot-instructions.md
    expected_artifacts = {
        'artifacts/model.pkl': None,
        'artifacts/vectorizer.pkl': None,
        'artifacts/detailed_model.pkl': None,
        'artifacts/vectorizer_detailed.pkl': None,
        'artifacts/toxicity_categories.pkl': None
    }
    
    print("=== Verificando artifacts en artifacts/ ===")
    for path in expected_artifacts:
        if os.path.exists(path):
            try:
                artifact = joblib.load(path)
                expected_artifacts[path] = artifact
                print(f"✅ {path}")
            except Exception as e:
                print(f"❌ {path} - Error loading: {e}")
        else:
            print(f"❌ {path} - No existe")
    
    # Verificar ubicación incorrecta
    print("\n=== Verificando artifacts/ (ubicación incorrecta) ===")
    wrong_location = [
        'artifacts/detailed_model.pkl',
        'artifacts/vectorizer_detailed.pkl', 
        'artifacts/toxicity_categories.pkl'
    ]
    
    found_in_wrong_location = []
    for path in wrong_location:
        if os.path.exists(path):
            print(f"🔄 {path} - ENCONTRADO (necesita moverse)")
            found_in_wrong_location.append(path)
        else:
            print(f"❌ {path}")
    
    # Test de funcionalidad
    basic_model = expected_artifacts['artifacts/model.pkl']
    basic_vectorizer = expected_artifacts['artifacts/vectorizer.pkl']
    detailed_model = expected_artifacts['artifacts/detailed_model.pkl']  # Nombre real
    detailed_vectorizer = expected_artifacts['artifacts/vectorizer_detailed.pkl']
    categories = expected_artifacts['artifacts/toxicity_categories.pkl']
    
    print("\n=== Test de predicción básica ===")
    if basic_model and basic_vectorizer:
        try:
            test_text = "You are an idiot"
            X_test = basic_vectorizer.transform([test_text])
            prediction = basic_model.predict(X_test)[0]
            proba = basic_model.predict_proba(X_test)[0]
            
            print(f"✅ Predicción básica OK")
            print(f"   Texto: '{test_text}'")
            print(f"   Predicción: {prediction} ({'Tóxico' if prediction else 'No tóxico'})")
            print(f"   Probabilidad: {proba}")
        except Exception as e:
            print(f"❌ Error en predicción básica: {e}")
    else:
        print("❌ Modelos básicos no disponibles")
    
    print("\n=== Test de predicción detallada ===")
    if detailed_model and detailed_vectorizer and categories:
        try:
            test_text = "You are an idiot"
            X_test = detailed_vectorizer.transform([test_text])
            prediction = detailed_model.predict(X_test)[0]
            probabilities = detailed_model.predict_proba(X_test)
            
            print(f"✅ Predicción detallada OK")
            print(f"   Categorías disponibles: {len(categories)}")
            print(f"   Predicción shape: {prediction.shape}")
            
            # Mostrar categorías detectadas
            detected = []
            for i, cat in enumerate(categories):
                if i < len(prediction) and prediction[i] == 1:
                    detected.append(cat)
            
            print(f"   Categorías detectadas: {detected[:3]}{'...' if len(detected) > 3 else ''}")
            
        except Exception as e:
            print(f"❌ Error en predicción detallada: {e}")
    else:
        print("❌ Modelos detallados no disponibles")
        if found_in_wrong_location:
            print("   Nota: Se encontraron artifacts en artifacts/")
    
    # Recomendaciones
    print("\n=== Recomendaciones ===")
    basic_ok = basic_model is not None and basic_vectorizer is not None
    detailed_ok = detailed_model is not None and detailed_vectorizer is not None and categories is not None
    
    if not basic_ok:
        print("❌ Ejecuta: jupyter notebook notebooks/03_guardar_modelo.ipynb")
    
    if not detailed_ok:
        if found_in_wrong_location:
            print("🔄 Artifacts detallados en ubicación incorrecta")
            print("   Solución 1: Mover manualmente a artifacts/")
            print("   Solución 2: Re-ejecutar notebook 04 con path correcto")
        else:
            print("❌ Ejecuta: jupyter notebook notebooks/04_modelo_clasificacion_detallada.ipynb")
            print("   Y asegúrate de guardar en artifacts/ (no artifacts/)")
    
    return basic_ok, detailed_ok

if __name__ == "__main__":
    basic_ok, detailed_ok = smoke_test_artifacts()
    
    if basic_ok and detailed_ok:
        print("\n🎉 SMOKE TEST EXITOSO")
        sys.exit(0)
    elif basic_ok:
        print("\n⚠️  SMOKE TEST PARCIAL - Solo básico funciona")
        sys.exit(1) 
    else:
        print("\n❌ SMOKE TEST FALLÓ")
        sys.exit(2)