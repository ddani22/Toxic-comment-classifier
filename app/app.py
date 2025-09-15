import streamlit as st
import joblib

clf = None
vectorizer = None
load_error = None

try:
    clf = joblib.load("artifacts/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer.pkl")
except Exception as e:
    load_error = e
    
if load_error is not None:
    st.exception(load_error)

st.title("🔎 Toxic Comment Classifier")

user_input = st.text_area("Escribe un comentario:")
if st.button("Clasificar"):
    if user_input.strip() != "":
        if vectorizer is not None and clf is not None:
            # Preprocesar y vectorizar
            X_new = vectorizer.transform([user_input])
            prediction = clf.predict(X_new)[0]
            result = "Tóxico ⚠️" if prediction == 1 else "No tóxico ✅"
            st.write(f"Resultado: {result}")
        else:
            st.error("El modelo o el vectorizador no están disponibles. No se puede procesar el comentario.")
    else:
        st.warning("Por favor escribe un comentario.")