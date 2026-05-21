#!/bin/bash

# Script pour exécuter le flux d'anonymisation à partir de la racine Graph-Anonymization

# Étape 1 : Aller dans le répertoire secGraph
#cd secGraph || { echo "Erreur : Impossible d'accéder au répertoire secGraph"; exit 1; }

# Étape 2 : Exécuter anonymizationscript.py
#python anonymizationscript.py || { echo "Erreur lors de l'exécution de anonymizationscript.py dans secGraph"; exit 1; }

# Étape 3 : Retourner à la racine
#cd ..

# Étape 4 : Aller dans le répertoire umga et générer le jar
cd umga || { echo "Erreur : Impossible d'accéder au répertoire umga"; exit 1; }
javac -cp ".:lib/*" -d bin -sourcepath src src/org/uoc/kison/Main.java
jar cfm UMGA.jar manifest.mf -C bin .

# Étape 5 : Exécuter pairstogml.py
python pairstogml.py || { echo "Erreur lors de l'exécution de pairstogml.py dans umga"; exit 1; }

# Étape 6 : Exécuter anonymizationscript.py dans umga
python anonymizationscript.py || { echo "Erreur lors de l'exécution de anonymizationscript.py dans umga"; exit 1; }

# Étape 7 : Retourner à la racine
cd ..

# Étape 8 : Exécuter gml_to_pairs_for_secgraph.py
python gml_to_pairs_for_secgraph.py || { echo "Erreur lors de l'exécution de gml_to_pairs_for_secgraph.py"; exit 1; }

# Étape 9 : Aller dans le répertoire kdld
cd kdld || { echo "Erreur : Impossible d'accéder au répertoire kdld"; exit 1; }

# Étape 10 : Exécuter pairs_to_networkx.py
python pairs_to_networkx.py || { echo "Erreur lors de l'exécution de pairs_to_networkx.py dans kdld"; exit 1; }

# Étape 11 : Exécuter anonymizationscript.py dans kdld
python anonymizationscript.py || { echo "Erreur lors de l'exécution de anonymizationscript.py dans kdld"; exit 1; }

# Étape 12 : Retourner à la racine
cd ..

# Étape 13 : Exécuter anonymizationscript.py à la racine
python anonymizationscript.py || { echo "Erreur lors de l'exécution de anonymizationscript.py à la racine"; exit 1; }

echo "Exécution terminée avec succès."