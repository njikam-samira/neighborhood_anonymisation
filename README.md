# Graph Anonymization - Ange + NCC

Ce projet implemente et evalue des methodes d'anonymisation de graphes, avec un focus sur la protection contre la re-identification des sommets.

## Contexte
L'anonymisation de graphes vise a publier un graphe utile tout en reduisant les risques de fuite d'information sur les noeuds.

Le projet part de la methode de **Ange** (basee principalement sur le degre) et propose une variante plus robuste.

## Probleme traite
Une anonymisation basee presque uniquement sur le degre peut rester vulnerable aux attaques de re-identification fondees sur la structure locale.

## Methode de base : Ange
Pipeline classique utilise ici :
1. construction de la sequence de degres ;
2. formation de clusters de taille au moins `k` ;
3. anonymisation intra-cluster (degre cible) ;
4. reconstruction du graphe anonymise ;
5. evaluation utilite / confidentialite.

## Limite de la methode Ange
La formation de clusters uniquement par degre peut regrouper des sommets structurellement differents, ce qui peut faciliter certaines attaques de re-identification.

## Contribution principale
La variante **Ange modifie NCC** integre, en plus du degre, des informations structurelles locales (NCC/signatures de voisinage) lors de la formation/harmonisation des groupes.

Objectif : former des groupes plus homogenes structurellement, ameliorer la confidentialite, tout en preservant mieux l'utilite du graphe anonymise.

## Etapes de l'approche
1. charger le graphe ;
2. calculer les proprietes des sommets (degre + NCC) ;
3. former des clusters de taille minimale `k` ;
4. anonymiser les sommets par groupe ;
5. reconstruire le graphe anonymise ;
6. evaluer confidentialite et utilite.

## Metriques utilisees
- Metriques de confidentialite : succes d'attaque de desanonymisation (SecGraph NS), verifications k-anonymity selon les methodes.
- Metriques structurelles : variation noeuds/aretes/densite, coefficient de clustering, APL.
- Metriques d'utilite applicative : prediction de liens (AUC, AP, Precision@k).
- Classification de noeuds : module GAT (si utilise dans vos experiences).

## Structure du projet
```text
k-Anonymization/
├── src/
│   └── graph_anonymization/
│       ├── anonymization/
│       │   ├── ange_original.py
│       │   ├── ange_modified.py
│       │   ├── zhou_pei.py
│       │   ├── reconstruction.py
│       │   └── reconstruction_optimize_mean.py
│       ├── attacks/
│       │   └── secgraph.py
│       ├── benchmarks/
│       │   ├── full_benchmark.py
│       │   └── hikda_benchmark.py
│       ├── data/
│       │   └── io.py
│       ├── evaluation/
│       │   ├── link_prediction.py
│       │   ├── run_link_prediction_experiment.py
│       │   ├── run_1hikda_link_prediction_experiment.py
│       │   ├── link_prediction_report.py
│       │   └── node_classification.py
│       └── metrics/
│           ├── structural_metrics.py
│           └── benchmark_metrics.py
├── experiments/
│   ├── run_full_benchmark.py
│   ├── run_link_prediction_experiment.py
│   ├── run_1hikda_benchmark.py
│   └── run_1hikda_link_prediction_experiment.py
├── data/
├── results/
├── legacy/
├── main.py
├── requirements.txt
└── .gitignore
```

## Installation
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Execution
### Option 1 - point d'entree unique
```bash
python main.py full-benchmark --datasets cora citeseer --k-values 2 5 8 10 15
python main.py link-prediction --input data/citeseer.pairs --k 10
python main.py hikda-benchmark --datasets cora citeseer --k-values 2 5 8 10
python main.py hikda-link-prediction --input data/citeseer.pairs --k 10
```

### Option 2 - scripts experiments
```bash
python experiments/run_full_benchmark.py --datasets cora citeseer --k-values 2 5 8 10 15
python experiments/run_link_prediction_experiment.py --input data/citeseer.pairs --k 10
python experiments/run_1hikda_benchmark.py --datasets cora citeseer --k-values 2 5 8 10
python experiments/run_1hikda_link_prediction_experiment.py --input data/citeseer.pairs --k 10
```

## Ajouter un nouveau jeu de donnees
1. ajouter `data/<nom_dataset>.pairs` (format : `u v` par ligne) ;
2. lancer un benchmark avec `--datasets <nom_dataset>` ;
3. verifier les sorties dans `results/`.

## Lecture des resultats
- `results/.../results_metrics.csv` : resultats structures + confidentialite.
- `results/.../comparison_report.pdf` : synthese PDF benchmark.
- `results/link_prediction/*.csv` : detail et resume prediction de liens.
- `results/link_prediction/*.png` : graphes comparatifs par methode.

## Notes importantes
- `legacy/` contient les anciens scripts de travail conserves pour tracabilite.
- Les wrappers a la racine (`run_full_benchmark.py`, `link_prediction_utility.py`, etc.) sont conserves pour compatibilite.
- Certaines experiences utilisent SecGraph (`secGraph/secGraphCLI.jar`) et Java.
