import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Cargar dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
df.columns = ['embarazos', 'glucosa', 'presion', 'pliegue', 'insulina', 'imc', 'funcion', 'edad', 'riesgo']

X = df.drop('riesgo', axis=1)
y = df['riesgo']

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Guardar modelo
joblib.dump(model, 'model.pkl')
