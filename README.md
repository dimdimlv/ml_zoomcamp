# ML Zoomcamp Homework 01

This repository contains the homework for Module 1 (Introduction) of the ML Zoomcamp.

## Contents

```
hw_01/
  car_fuel_efficiency.csv   # Dataset used in the notebook
  hw_01.ipynb               # Jupyter notebook with the homework solutions / exploration
```

## Dataset
`car_fuel_efficiency.csv` appears to contain automobile characteristics and corresponding fuel efficiency (MPG). The notebook performs basic EDA and model preparation (adjust as needed — feel free to expand this section after pushing).

## Getting Started

### Prerequisites
- Python 3.9+ (any recent version should work)
- (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows: .venv\\Scripts\\activate
```

### Install common packages (if you later add code that needs them)
If you add model training code, you will likely need packages such as:
```bash
pip install pandas numpy scikit-learn jupyter matplotlib seaborn
```

### Open the Notebook
```bash
jupyter notebook hw_01/hw_01.ipynb
```
(or use VS Code / PyCharm built-in notebook support.)

## How to Reproduce
1. Clone the repository after you push it to GitHub:
   ```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
   ```
2. Create a virtual environment (optional but recommended).
3. Open the notebook and execute cells in order.

## Git / GitHub Workflow
Typical flow once this repo is on GitHub:
```bash
git pull origin main            # Update local
# ... make changes ...
git add .                        # Stage changes
git commit -m "Describe changes"
git push origin main             # Push
```

## Next Steps / Ideas
- Add a lightweight baseline regression model (e.g., LinearRegression) with train/validation split.
- Add a `requirements.txt` once dependencies stabilize.
- Include a brief EDA summary in the README.
- Add Makefile or simple script to reproduce preprocessing steps.

## License
Add a license of your choice (e.g., MIT) if you plan to share or expand this repository.

---
Feel free to adjust this README after the initial push.
