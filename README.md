# hackathon_solution
# BigMart Sales Prediction

## 📌 Project Overview
This project predicts the sales of products at various BigMart outlets using historical sales data.  
The dataset contains product and store attributes, and the goal is to build a regression model to forecast `Item_Outlet_Sales` for unseen data.

---

## 📂 Repository Structure

---

## 🛠 Steps Performed
1. **EDA**
   - Missing value analysis
   - Distribution plots & correlations
   - Outlet and product-level insights
2. **Feature Engineering**
   - Missing value imputation
   - Categorical encoding
   - Feature transformations
3. **Modeling**
   - Trained a LightGBM regressor
   - Hyperparameter tuning
4. **Prediction**
   - Generated `submission.csv` in required format

---

## 📊 Dataset
The dataset includes:
- **Train:** 8523 rows with `Item_Outlet_Sales` as target
- **Test:** 5681 rows for prediction

Key columns:
- `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`
- `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`

---

## 🚀 How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm

# Run the main script
python bigmart_solution.py
