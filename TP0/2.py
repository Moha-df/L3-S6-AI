import pandas as pd
import numpy as np

def manipulation_dataframe():
    print("Manipulation du DataFrame Abalone")
    print("-" * 50)
    
    # 1. Importer les données
    df = pd.read_csv("abalone.csv")
    
    # 2. Afficher les 10 premières instances
    print("10 premières instances:")
    print(df.head(10))
    
    # 3. Afficher les noms des attributs
    print("\nNoms des attributs:")
    print(df.columns.tolist())
    
    # 4. Nombre d'instances et vérification des valeurs manquantes
    print("\nNombre d'instances:", df.shape[0])
    print("Valeurs manquantes:")
    print(df.isnull().sum())
    
    # 5. Types des attributs
    print("\n\n\nTypes des attributs:")
    print(df.dtypes)
    
    # 6. Analyse des anneaux
    print("\nAnalyse des anneaux:")
    print("Nombre de valeurs différentes:", df['Rings'].nunique())
    print("Top 3 valeurs les plus représentées:")
    print(df['Rings'].value_counts().head(3))
    
    # 7. Analyse du sexe
    print("\nAnalyse du sexe:")
    print("Valeurs possibles:", df['Sex'].unique())
    print("Nombre d'instances par valeur:")
    print(df['Sex'].value_counts())
    
    # 7(c) Vérification de l'équilibre des classes
    print("Répartition équilibrée ?", df['Sex'].value_counts(normalize=True))
    
    # 7(d) Transformation des valeurs de 'Sex' en codes entiers
    df['Sex'] = df['Sex'].astype('category').cat.codes
    
    # 7(e) Séparation des attributs et cibles
    X = df.drop(columns=['Sex'])
    y = df['Sex']
    
    # 7(f) Conversion en tableaux numpy
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    print("X (Attributs) :")
    print(X.head())  # Affiche les 5 premières lignes de X

    print("\ny (Cible) :")
    print(y.head())  # Affiche les 5 premières valeurs de y
    
    print("\nTransformation terminée.\n")
    return X_np, y_np

if __name__ == "__main__":
    X, y = manipulation_dataframe()