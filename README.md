# FE4MT
Ce projet vise à construire un système permettant d'explorer les données multitables afin d'en extraire les paramètres pertinents pour la classification. Les paramètres pris en compte ici sont les variables natives secondaires (variables présentes dans les tables secondaires) et les primitives de construction (règles mathématiques).

Ce système s'est dans un premier temps concentré sur la création d'une mesure d'importance des variables et des primitives vis-à-vis de la variable cible. L'objectif final sera ainsi de réduire l'espace des primitives et des variables pour sélectionner uniquement les éléments pertinents.

## Méthode employée
La méthode d'estimation d'importance employée est une méthode univariée avec discrétisation de la variable Count pour limiter l'effet du bruit sur les tables secondaires. Les explications et les différentes expérimentations ayant mené au choix de cette méthode sont présentées dans le rapport de stage [Rapport_Stage_Lou-Anne_Quellet](Rapport_Stage_Lou-Anne_Quellet.pdf).

## Environnement
Le projet a été effectué via l'environnement conda `environnement.yml`. L'environnement est téléchargeable via la commande :
```bash
conda env create --file=environnement.yml
```

## Organisation
Les dossiers sont composés de :
* DATA : Un jeu de données : Accident (possède son propre README)
* Etude : le code python a exécuter.
* function : les programmes comportant la classe et les fonctions utiles pour les études


