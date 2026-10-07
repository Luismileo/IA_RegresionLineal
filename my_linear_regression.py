# my_linear_regression.py
import numpy as np

class MyLinearRegression:
    def __init__(self):
        # Atributos exigidos en la consigna
        self.coef_ = None      # Parámetros / Pesos de las variables
        self.intercept_ = None # Término independiente (\theta_0)

    def fit(self, X, y):
        """
        Ajusta el modelo a los datos de entrenamiento usando la Ecuación Normal:
        \theta = (X^T * X)^(-1) * X^T * y
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # 1. Agregar columna de 1s a X para representar \theta_0 (intercepto)
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # 2. Aplicar la Ecuación Normal
        # (X_b.T @ X_b) calcula el producto X^T * X
        # np.linalg.inv() calcula la inversa
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        # 3. Guardar los parámetros en los atributos requeridos
        self.intercept_ = float(theta[0][0])
        self.coef_ = theta[1:].ravel()
        return self

    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos.
        y_pred = X * coef_ + intercept_
        """
        X = np.array(X)
        return X @ self.coef_ + self.intercept_

    def mse(self, y_true, y_pred):
        """
        Calcula el Error Cuadrático Medio (MSE).
        """
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        return float(np.mean((y_true - y_pred) ** 2))

    def r2_score(self, y_true, y_pred):
        """
        Calcula el Coeficiente de Determinación (R^2).
        """
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        ss_res = np.sum((y_true - y_pred) ** 2)          # Suma de residuos al cuadrado
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # Suma total de cuadrados
        
        return float(1 - (ss_res / ss_tot))