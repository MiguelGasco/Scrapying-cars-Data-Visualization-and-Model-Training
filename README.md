# Scrapy Data Pipeline Project

**Puedes ver los gráficos de este proyecto en el siguiente enlace :** 

https://scrapying-cars-data-visualization-and-model-training-aeemtle6s.streamlit.app

Este proyecto utiliza Scrapy para realizar scraping de una web, seguido de un proceso de tratamiento y limpieza de los datos extraídos. Finalmente, los datos procesados se visualizan y se utilizan para entrenar un modelo de machine learning.

## Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Requisitos](#requisitos)

## Descripción del Proyecto

Este proyecto automatiza la recolección de datos desde un sitio web específico, procesa los datos eliminando inconsistencias y valores nulos, los visualiza para obtener información clave y, finalmente, entrena un modelo de aprendizaje automático utilizando los datos limpios.

El flujo completo del proyecto incluye:
- **Scraping:** Recolección de datos utilizando Scrapy.
- **Transformación y Limpieza de Datos:** Uso de Pandas y otras librerias para limpiar y organizar los datos.
- **Visualización de Datos:** Uso de Matplotlib, Seaborn y Plotly Express para visualizar patrones y tendencias.
- **Entrenamiento del Modelo:** Uso de Scikit-learn para entrenar un modelo de aprendizaje automático.

## Requisitos

Este proyecto requiere las siguientes herramientas y bibliotecas:

- scrapy
- pandas
- numpy
- scikit-learn
- streamlit
- plotly
- matplotlib
- seaborn
- joblib

Instala los requisitos ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
