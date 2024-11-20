# Charger les données
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Charger le fichier Excel
data_titanic = pd.read_excel("C:\\Users\\HP\\Downloads\\titanic3.xlsx", engine='openpyxl')


print(data_titanic.head())

# Préparer les données pour l'arbre de décision
X = data_titanic[['pclass', 'sex', 'age', 'fare']]  # Utilisation des colonnes pertinentes
X['sex'] = X['sex'].map({'male': 0, 'female': 1})  # Encodage de la colonne 'sex'

# Gérer les valeurs manquantes
X['age'] = X['age'].fillna(X['age'].median())
X['fare'] = X['fare'].fillna(X['fare'].median())

# La colonne cible (label)
y = data_titanic['survived']

# Créer et entraîner l'arbre de décision
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X, y)

# Afficher et enregistrer l'arbre de décision
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Arbre de Décision - Titanic Survival")

# Enregistrer l'image
plt.savefig("arbre_decision_titanic.png", dpi=300, bbox_inches='tight')  # Sauvegarde en PNG
plt.show()
