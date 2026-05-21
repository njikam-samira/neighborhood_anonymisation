#!/bin/bash

# Script pour exécuter le flux d'utilité à partir de la racine Graph-Anonymization

# Étape 1 : Aller dans le répertoire secGraph
cd secGraph || { echo "Erreur : Impossible d'accéder au répertoire secGraph"; exit 1; }

# Étape 2 : utilityscript.py
python utilityscript.py || { echo "Erreur lors de l'exécution de utilityscript.py dans secGraph"; exit 1; }

echo "Exécution terminée avec succès."