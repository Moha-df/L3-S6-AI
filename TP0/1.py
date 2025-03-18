import numpy as np

def exercice_1():
    print("Exercice 1: Création et manipulation d'un vecteur")
    print("-" * 50)
    
    # 1. Créer un vecteur v de 10 éléments entiers aléatoires de 0 à 10
    v = np.random.randint(0, 11, 10)
    print(f"Vecteur v créé: {v}")
    
    # (a) Sélectionner tous les éléments de ce vecteur jusqu'aux 4ème (inclus)
    print(f"(a) Éléments jusqu'au 4ème inclus: {v[:5]}")
    
    # (b) Sélectionner tous les éléments de ce vecteur après le 4ème
    print(f"(b) Éléments après le 4ème: {v[5:]}")
    
    # (c) Sélectionner le dernier élément de ce vecteur
    print(f"(c) Dernier élément: {v[-1]}")
    
    # (d) Sélectionner tous les éléments de ce vecteur sauf le dernier
    print(f"(d) Tous les éléments sauf le dernier: {v[:-1]}")
    
    print("\n")
    return v

def exercice_2():
    print("Exercice 2: Création et manipulation d'une matrice")
    print("-" * 50)
    
    # 2. Créer une matrice a de 4×3 éléments entiers aléatoires de 0 à 10 inclus
    a = np.random.randint(0, 11, (4, 3))
    print(f"Matrice a créée:\n{a}")
    
    # (a) Sélectionner les 2 premières ligne de la matrice
    print(f"(a) Les 2 premières lignes:\n{a[:2, :]}")
    
    # (b) Sélectionner les 2 premières colonnes de la matrice
    print(f"(b) Les 2 premières colonnes:\n{a[:, :2]}")
    
    # (c) Sélectionner les 2 premières lignes de la 3ème colonne de la matrice
    print(f"(c) Les 2 premières lignes de la 3ème colonne:\n{a[:2, 2]}")
    
    # (d) Sélectionner le dernier élément de la matrice
    print(f"(d) Le dernier élément: {a[-1, -1]}")
    
    # (e) Afficher la forme de la matrice
    print(f"(e) Forme de la matrice: {a.shape}")
    
    # (f) Afficher le nombre de lignes de la matrice
    print(f"(f) Nombre de lignes: {a.shape[0]}")
    
    # (g) Afficher le nombre de colonnes de la matrice
    print(f"(g) Nombre de colonnes: {a.shape[1]}")
    
    # (h) Afficher le nombre total d'éléments dans la matrice
    print(f"(h) Nombre total d'éléments: {a.size}")
    
    # (i) Afficher la dimension de la matrice
    print(f"(i) Dimension de la matrice: {a.ndim}")
    
    print("\n")
    return a

def exercice_3(v, a):
    print("Exercice 3: Manipulation avancée de vecteurs et matrices")
    print("-" * 50)
    
    # (a) Transposer le vecteur précédemment créé
    v_transpose = v.reshape(1, -1)  # Reshape pour rendre la transposition significative
    print(f"(a) Vecteur v transposé:\n{v_transpose}")
    
    # (b) Aplatir la matrice précédemment créée
    a_aplatie = a.flatten()
    print(f"(b) Matrice a aplatie: {a_aplatie}")
    print("   Le vecteur contient les éléments ligne par ligne, de gauche à droite")
    
    # (c) Redimensionner v de façon à l'afficher comme une matrice de 2 lignes et 5 colonnes
    v_reshaped = v.reshape(2, 5)
    print(f"(c) Vecteur v redimensionné:\n{v_reshaped}")
    
    # (d) Créer une seconde matrice m avec des valeurs aléatoires distribuées normalement
    np.random.seed(42)  # Fixer la graine du générateur
    m = np.random.normal(2, 1, a.shape)
    print(f"(d) Matrice m créée (distribution normale, moyenne=2, écart-type=1):\n{m}")
    
    # (e) Est-ce possible de multiplier les matrices a et m ?
    print("(e) Est-ce possible de multiplier les matrices a et m ?")
    print("   Non, car pour multiplier deux matrices, le nombre de colonnes de la première")
    print("   doit être égal au nombre de lignes de la seconde. Ici, les deux matrices")
    print(f"   ont la même forme: {a.shape}")
    
    # (f) Créer la matrice m2 qui sera la transposée de m
    m2 = m.T
    print(f"(f) Matrice m2 (transposée de m):\n{m2}")
    
    # (g) Multiplier les matrices a et m2
    try:
        a_m2 = np.dot(a, m2)
        print(f"(g) Multiplication de a et m2:\n{a_m2}")
    except ValueError as e:
        print(f"(g) Erreur: {e}")
    
    # (h) Calculer la moyenne et l'écart type de m2
    mean_m2 = np.mean(m2)
    std_m2 = np.std(m2)
    print(f"(h) Moyenne de m2: {mean_m2}")
    print(f"    Écart type de m2: {std_m2}")
    
    # (i) Créer deux matrices carrées a1 et a2
    a1 = np.random.randint(0, 10, (3, 3))
    a2 = np.random.randint(0, 10, (3, 3))
    print(f"(i) Matrice a1:\n{a1}")
    print(f"    Matrice a2:\n{a2}")
    
    # (j) Sommer a1 et a2
    somme = a1 + a2
    print(f"(j) Somme de a1 et a2:\n{somme}")
    
    # (k) Soustraire a1 et a2
    difference = a1 - a2
    print(f"(k) Différence entre a1 et a2:\n{difference}")


def main():
    # Exécuter les trois exercices séquentiellement
    #print("MANIPULATION DE VECTEURS ET MATRICES AVEC NUMPY")
    #print("=" * 60)
    
    # Exercice 1
    v = exercice_1()
    
    # Exercice 2
    a = exercice_2()
    
    # Exercice 3
    exercice_3(v, a)


if __name__ == "__main__":
    main()