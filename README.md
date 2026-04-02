# Neural Network - Logistic Regression TP

##  Description

Ce projet implémente une **régression logistique avec une perspective réseau de neurones** pour la classification d'images (chat vs non-chat).

C'est un TP (Travaux Pratiques) pédagogique couvrant :
- ✅ Initialisation des paramètres
- ✅ Forward et Backward propagation
- ✅ Optimisation par gradient descent
- ✅ Prédiction et évaluation du modèle

---

##  Objectifs

Construire un algorithme simple de reconnaissance d'images capable de :
- Distinguer un chat d'une non-chat
- Atteindre une précision de ~70% sur l'ensemble de test
- Démontrer les fondamentaux des réseaux de neurones

---

##  Structure du projet

```
Neural_network/
├── README.md                           (cette documentation)
├── Logistic (2).ipynb                  (notebook principal)
├── lr_utils.py                         (utilitaires)
├── datasets/
│   ├── train_catvnoncat.h5             (données d'entraînement)
│   └── test_catvnoncat.h5              (données de test)
└── images/                             (images pour test)
    ├── LogReg_kiank.png
    ├── cat_in_iran.jpg
    ├── image1.png
    ├── image2.png
    └── ... (autres images de test)
```

---

## Installation & Utilisation

### Prérequis

```bash
pip install numpy matplotlib h5py scipy pillow
```

### Exécution du notebook

1. Ouvrez `Logistic (2).ipynb` dans Jupyter Notebook ou VS Code
2. Exécutez les cellules dans l'ordre
3. Le modèle s'entraînera automatiquement sur les données

### Tester avec votre propre image

1. Placez votre image dans le dossier `images/`
2. Modifiez la variable dans la dernière cellule :
   ```python
   my_image = "votre_image.jpg"   # changez le nom
   ```
3. Exécutez la cellule pour voir la prédiction

---

##  Résultats attendus

- **Train Accuracy** : ~99%
- **Test Accuracy** : ~70%
- **Prédiction** : 1 = Chat, 0 = Non-Chat

---

##  Fonctions principales

| Fonction | Description |
|----------|-------------|
| `sigmoid(z)` | Activation sigmoid |
| `initialize_with_zeros(dim)` | Initialisation des poids et biais |
| `propagate(w, b, X, Y)` | Forward + Backward propagation |
| `optimize(w, b, X, Y, ...)` | Gradient descent optimization |
| `predict(w, b, X)` | Prédiction sur nouvelles données |
| `model(X_train, Y_train, ...)` | Pipeline complet |

---

## Analyse du learning rate

Le notebook inclut une analyse comparative de différents learning rates :
- **0.01** : Trop grand (oscillation)
- **0.001** : Optimal
- **0.0001** : Trop petit (convergence lente)

---

##  Formules mathématiques

### Forward Propagation
$$z = w^T X + b$$
$$A = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### Cost Function
$$J = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(a^{(i)}) + (1-y^{(i)}) \log(1-a^{(i)})]$$

### Gradients
$$\frac{\partial J}{\partial w} = \frac{1}{m} X(A-Y)^T$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)} - y^{(i)})$$

### Update Rule
$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

---

##  Notes pédagogiques

- Importance de la **normalisation** des données (division par 255)
- Choix du **learning rate** crucial pour la convergence
- **Overfitting** visible (train 99% vs test 70%)
- Logistic Regression = réseau de neurones à 1 couche

---

##  Ressources

- [Implementing Neural Networks from Scratch](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
- [Image Normalization - Stack Exchange](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)

---

##  Auteur

**Salma Elhilaly**  
Email : salma.elhilaly1@uit.ac.ma
