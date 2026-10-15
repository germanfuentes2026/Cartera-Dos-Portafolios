# Frontera eficiente AAPL - TSLA

Aplicación Streamlit para calcular y visualizar 100 portafolios long-only
de Apple (AAPL) y Tesla (TSLA) con diez años de precios diarios ajustados
descargados de Yahoo Finance.

## Archivos
- `app.py`: aplicación completa.
- `requirements.txt`: dependencias.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud
1. Subí `app.py` y `requirements.txt` a un repositorio de GitHub.
2. En https://share.streamlit.io/ elegí el repositorio.
3. Seleccioná `app.py` como archivo principal y desplegá.

Los retornos se anualizan multiplicando la media diaria por 252 y la
covarianza diaria por 252. Los resultados son estimaciones históricas.
