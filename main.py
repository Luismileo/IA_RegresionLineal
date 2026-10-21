import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Importa tu modelo desde el archivo local
from my_linear_regression import MyLinearRegression

# 1. Carga los datos limpios
df = pd.read_csv("HousingInspected.csv")

# 2. Separa las variables predictoras (X) y el precio (y)
X = df[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y = df["price"]

# 3. Divide los datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entrena TU modelo
my_model = MyLinearRegression()
my_model.fit(X_train, y_train)

y_pred_my = my_model.predict(X_test)
mse_my = my_model.mse(y_test, y_pred_my)
r2_my = my_model.r2_score(y_test, y_pred_my)

# 5. Entrena el modelo de SCIKIT-LEARN
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

y_pred_sk = sk_model.predict(X_test)
mse_sk = mean_squared_error(y_test, y_pred_sk)
r2_sk = r2_score(y_test, y_pred_sk)

# 6. Muestra la comparación en pantalla
print("=== MI MODELO ===")
print("Intercepto:", my_model.intercept_)
print("Coeficientes:", my_model.coef_)
print("MSE:", mse_my)
print("R2:", r2_my)

print("\n=== MODELO SCIKIT-LEARN ===")
print("Intercepto:", sk_model.intercept_)
print("Coeficientes:", sk_model.coef_)
print("MSE:", mse_sk)
print("R2:", r2_sk)