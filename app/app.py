import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Variables globales para modelos
clf = None
vectorizer = None
detailed_clf = None
detailed_vectorizer = None
toxicity_categories = None
load_error = None

# Cargar modelo básico
try:
    clf = joblib.load("artifacts/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer.pkl")
    st.success("✅ Modelo básico cargado correctamente")
except Exception as e:
    load_error = e
    st.error(f"❌ Error cargando modelo básico: {e}")

# Cargar modelo detallado (opcional)
detailed_available = False
try:
    detailed_clf = joblib.load("artifacts/detailed_model.pkl")
    detailed_vectorizer = joblib.load("artifacts/vectorizer_detailed.pkl")
    toxicity_categories = joblib.load("artifacts/toxicity_categories.pkl")
    detailed_available = True
    st.success("✅ Modelo de clasificación detallada disponible")
except Exception as e:
    st.warning(f"⚠️ Clasificación detallada no disponible: {e}")

# Configuración de la página
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Clasificador de Comentarios Tóxicos")
st.markdown("---")

# Sidebar con información del modelo
with st.sidebar:
    st.header("📊 Estado del Sistema")
    st.write(f"**Modelo básico:** {'✅ Activo' if clf is not None else '❌ Error'}")
    st.write(f"**Clasificación detallada:** {'✅ Disponible' if detailed_available else '❌ No disponible'}")
    
    if detailed_available:
        if toxicity_categories is not None:
            st.write(f"**Categorías detectables:** {len(toxicity_categories)}")
        if detailed_vectorizer is not None and hasattr(detailed_vectorizer, 'vocabulary_'):
            st.write(f"**Vocabulario:** {len(detailed_vectorizer.vocabulary_):,} palabras")
        else:
            st.write("**Vocabulario:** No disponible")

# Input del usuario
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Escribe un comentario para analizar:",
        height=100,
        placeholder="Ejemplo: Este es un comentario de prueba..."
    )

with col2:
    st.markdown("### Opciones de análisis")
    show_probability = st.checkbox("Mostrar probabilidades", value=True)
    show_detailed = st.checkbox(
        "Análisis detallado", 
        value=detailed_available,
        disabled=not detailed_available
    )

# Botón de clasificación
if st.button("🔍 Clasificar Comentario", type="primary"):
    if user_input.strip() != "":
        if vectorizer is not None and clf is not None:
            try:
                # Clasificación básica
                X_new = vectorizer.transform([user_input])
                prediction = clf.predict(X_new)[0]
                is_toxic = bool(prediction)
                
                # Mostrar resultado principal
                if is_toxic:
                    st.error("🚨 **COMENTARIO TÓXICO DETECTADO**")
                else:
                    st.success("✅ **Comentario No Tóxico**")
                
                # Mostrar probabilidades básicas
                if show_probability:
                    prob_scores = clf.predict_proba(X_new)[0]
                    prob_toxic = prob_scores[1] if len(prob_scores) > 1 else 0.0
                    prob_safe = prob_scores[0] if len(prob_scores) > 1 else 1.0
                    
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Probabilidad Tóxico", f"{prob_toxic:.1%}")
                    with col_prob2:
                        st.metric("Probabilidad No Tóxico", f"{prob_safe:.1%}")
                
                # Análisis detallado (solo si es tóxico y está disponible)
                if (show_detailed and is_toxic and detailed_available and 
                    detailed_clf is not None and detailed_vectorizer is not None and 
                    toxicity_categories is not None):
                    st.markdown("---")
                    st.subheader("🔬 Análisis Detallado de Toxicidad")
                    
                    try:
                        # Clasificación detallada
                        X_detailed = detailed_vectorizer.transform([user_input])
                        detailed_predictions = detailed_clf.predict(X_detailed)[0]
                        detailed_probabilities = detailed_clf.predict_proba(X_detailed)
                        
                        # Preparar datos para visualización
                        categories_data = []
                        
                        for i, category in enumerate(toxicity_categories):
                            if i < len(detailed_predictions):
                                # Obtener probabilidad
                                prob = 0.0
                                if i < len(detailed_probabilities):
                                    proba_array = detailed_probabilities[i][0]
                                    if len(proba_array) > 1:
                                        prob = float(proba_array[1])
                                
                                is_present = bool(detailed_predictions[i])
                                
                                if is_present or prob > 0.1:  # Mostrar solo relevantes
                                    categories_data.append({
                                        'Categoría': category.replace('_', ' ').title(),
                                        'Detectado': '✅' if is_present else '❌',
                                        'Probabilidad': f"{prob:.1%}",
                                        'Valor_Prob': prob
                                    })
                        
                        if categories_data:
                            # Ordenar por probabilidad
                            categories_data.sort(key=lambda x: x['Valor_Prob'], reverse=True)
                            
                            # Mostrar en columnas
                            col_det1, col_det2 = st.columns(2)
                            
                            # Categorías detectadas
                            detected = [cat for cat in categories_data if cat['Detectado'] == '✅']
                            if detected:
                                with col_det1:
                                    st.markdown("**🎯 Categorías Detectadas:**")
                                    for cat in detected:
                                        st.write(f"• **{cat['Categoría']}** ({cat['Probabilidad']})")
                            
                            # Categorías con probabilidad alta pero no detectadas
                            suspicious = [cat for cat in categories_data if cat['Detectado'] == '❌' and cat['Valor_Prob'] > 0.3]
                            if suspicious:
                                with col_det2:
                                    st.markdown("**⚠️ Posibles Indicadores:**")
                                    for cat in suspicious:
                                        st.write(f"• {cat['Categoría']} ({cat['Probabilidad']})")
                            
                            # Tabla detallada (colapsible)
                            with st.expander("📋 Ver tabla completa de análisis"):
                                df_results = pd.DataFrame([
                                    {k: v for k, v in cat.items() if k != 'Valor_Prob'} 
                                    for cat in categories_data
                                ])
                                st.dataframe(df_results, use_container_width=True)
                        else:
                            st.info("No se detectaron categorías específicas de toxicidad.")
                            
                    except Exception as detail_error:
                        st.error(f"Error en análisis detallado: {detail_error}")
                
                elif show_detailed and not is_toxic:
                    st.info("ℹ️ El análisis detallado solo se realiza para comentarios clasificados como tóxicos.")
                
                # Timestamp
                st.caption(f"Análisis realizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"Error durante la clasificación: {e}")
        else:
            st.error("❌ El modelo o el vectorizador no están disponibles.")
    else:
        st.warning("⚠️ Por favor escribe un comentario para analizar.")

# Sección de ejemplos
st.markdown("---")
with st.expander("💡 Ejemplos de comentarios para probar"):
    st.markdown("""
    **Comentarios no tóxicos:**
    - "Me gusta mucho este artículo, muy informativo."
    - "Gracias por compartir tu opinión, es muy interesante."
    
    **Comentarios potencialmente tóxicos:**
    - "Eres un idiota por pensar eso."
    - "Tu opinión es basura y no vale nada."
    
    *Nota: Estos ejemplos son solo para fines de demostración.*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🔎 Toxic Comment Classifier | Powered by Machine Learning"
    "</div>", 
    unsafe_allow_html=True
)