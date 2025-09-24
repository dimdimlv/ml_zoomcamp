# ML Zoomcamp Homework 01

This repository contains the homework for Module 1 (Introduction) of the ML Zoomcamp.

## Contents

```
hw_01/
  car_fuel_efficiency.csv   # Dataset used in the notebook
  hw_01.ipynb               # Jupyter notebook with the homework solutions / exploration
```

## Dataset
`car_fuel_efficiency.csv` appears to contain automobile characteristics and corresponding fuel efficiency (MPG). The notebook performs basic EDA and model preparation 

## Getting Started

### Anaconda and Conda

The easiest way to set up the environment is to use [Anaconda](https://www.anaconda.com/products/individual) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Anaconda comes with everything we need (and much more). 
Miniconda is a smaller version of Anaconda that contains only Python. 

Follow the instructions on page for installing the correct package for your system.
The site will automatically detect your operating system and suggest the correct package.

* [Anaconda](https://www.anaconda.com/products/individual)
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)

Anaconda is recommended.


### (Optional) Create environment for course

It is a good idea to set up a dedicated environment for the course 

In your terminal, run this command to create the environment

```bash
conda create -n ml-zoomcamp python=3.11
```

Activate it:

```bash
conda activate ml-zoomcamp
```

Installing libraries

```bash
conda install numpy pandas scikit-learn seaborn jupyter
```

### Open the Notebook
```bash
jupyter notebook hw_01/hw_01.ipynb
```
(or use VS Code / PyCharm built-in notebook support.)
