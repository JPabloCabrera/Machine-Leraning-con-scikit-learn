# Machine Learning Professional Course with scikit-learn

Welcome to the "Curso Profesional de Machine Learning con scikit-learn" project! This project encompasses a series of hands-on exercises and projects undertaken as part of the professional machine learning course at Platzi. The course provides an in-depth exploration of machine learning concepts using the scikit-learn library in Python.

## Overview

**Duration:** 4 Hours of Content | 16 Hours of Practice

**Instructor:** Ariel Ortiz, COO at Spin Quantum Tech

## Course Highlights

- Implementation of Robust Regression using the Happiness Index dataset.
- Key concepts in machine learning and scikit-learn.
- Feature optimization and Principal Component Analysis (PCA).
- Regularization techniques: Lasso, Ridge, and ElasticNet.
- Introduction to ensemble methods: Bagging and Boosting.
- Clustering strategies and implementation using K-Means and Mean-Shift.
- Parametric optimization with Cross Validation.
- Randomized optimization and an introduction to Auto Machine Learning.
- Deployment considerations and creating an API with Flask.

## Project Descriptions

### Project 1: Robust Regression with the Happiness Index Dataset

This project focuses on implementing a robust regression model using the Happiness Index dataset. The objective is to clean and transform data outliers for robust regression processing.

### Project 2: Ensemble Methods and Clustering

The second project delves into ensemble methods and clustering using datasets such as Pima Indians Diabetes and Car Evaluation. The goal is to classify diabetes presence and assess car quality based on various features.

### Project 3: Deployment and API Creation

The final project involves revisiting code architecture, importing/exporting models with scikit-learn, and creating an API with Flask for the deployed model.

## License

These projects are shared under the MIT license. For more details, refer to the [LICENSE](LICENSE) file.

## Contact

Instructor: Ariel Ortiz
School: Platzi School of Data Science and Artificial Intelligence

Thank you for exploring these projects that cover decision trees, random forests, and classification concepts using scikit-learn in various datasets!

cursoprofesionalparamachinelearningconsklearn
  
## Installation guide

Please read [install.md](install.md) for details on how to set up this project.

## Project Organization

    ├── LICENSE
    ├── tasks.py           <- Invoke with commands like `notebook`.
    ├── README.md          <- The top-level README for developers using this project.
    ├── install.md         <- Detailed instructions to set up this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting.
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment.
    │
    ├── .here              <- File that will stop the search if none of the other criteria
    │                         apply when searching head of project.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .)
    │                         so profetional_ml_sklearn can be imported.
    │
    └── profetional_ml_sklearn               <- Source code for use in this project.
        ├── __init__.py    <- Makes profetional_ml_sklearn a Python module.
        │
        ├── data           <- Scripts to download or generate data.
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions.
        │   ├── predict_model.py
        │   └── train_model.py
        │
        ├── utils          <- Scripts to help with common tasks.
            └── paths.py   <- Helper functions to relative file referencing across project.
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations.
            └── visualize.py

---
