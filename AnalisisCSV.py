# Análisis exploratorio y limpieza del dataset antes del modelo.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset original
df_original = pd.read_csv('Housing.csv')

# 2. Seleccionar variables cuantitativas/continuas
columnas_num = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
df_inspected = df_original[columnas_num].copy()

# 3. Limpieza de datos (dropna elimina valores nulos)
df_inspected = df_inspected.dropna()

# 4. Guardar el nuevo dataset limpio
df_inspected.to_csv('HousingInspected.csv', index=False)

print("¡Archivo 'HousingInspected.csv' creado con éxito!")
print(f"Dimensiones: {df_inspected.shape[0]} ejemplos y {df_inspected.shape[1]} columnas.")

# --- ANÁLISIS EXPLORATORIO (Exigido en el Punto 1) ---
print("\n--- RESUMEN ESTADÍSTICO ---")
print(df_inspected.describe())

# Visualización: Matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(df_inspected.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación de Variables Numéricas")
plt.tight_layout()
plt.show()