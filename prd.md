# 📋 Product Requirements Document (PRD)
# StatViz Pro — Free Web-Based Statistical Analysis Platform
### Inspired by IBM SPSS | Built for Data Scientists & Researchers

---

## 1. 🎯 Project Overview

**Product Name:** StatViz Pro
**Tagline:** "Your Free, Browser-Based SPSS Alternative"
**Target Users:** Students, researchers, data scientists, MPhil/PhD scholars, statisticians (especially in Pakistan)
**Primary Reference:** IBM SPSS + Codanics ABC of Statistics Book
**Deployment:** Free-tier cloud platforms (Streamlit Community Cloud / Hugging Face Spaces)

---

## 2. 📱 Platform and Accessibility

| Platform | Features Available |
|---|---|
| Desktop Browser | Full access: Upload → EDA → Plots → Modeling |
| Mobile Browser | Upload dataset, EDA, Descriptive Stats, basic Plots |
| Tablet Browser | Full access (responsive layout) |

---

## 3. 🗂️ Tech Stack (Free-Friendly)

| Layer | Technology | Reason |
|---|---|---|
| Frontend/UI | Streamlit | Free hosting on Streamlit Community Cloud, Python-native |
| Data Processing | Pandas, NumPy | Industry standard, free |
| Interactive Plots | Plotly | Interactive, supports PNG/SVG/JPEG export |
| Static Plots | Matplotlib, Seaborn | Advanced customization |
| Machine Learning | Scikit-learn | Free, comprehensive |
| Statistical Tests | SciPy, Statsmodels | Full stats support |
| File Format Support | openpyxl, xlrd | Excel support |
| Deployment | Streamlit Community Cloud | 100% Free |
| Backup Deployment | Hugging Face Spaces | Free, mobile-accessible |

---

## 4. 🏗️ Application Architecture (Module Map)

```
StatViz Pro
│
├── Home Page
│   ├── App Introduction
│   ├── How to Use Guide
│   └── Quick Start Button
│
├── Data Import Module
│   ├── Upload CSV, XLSX, XLS, JSON, TXT
│   ├── Preview Dataset (first N rows)
│   ├── Dataset Shape (rows x columns)
│   ├── Column Data Types Detection
│   └── Variable Classification (Numeric / Categorical / DateTime)
│
├── Data Type Management & Transformation Module (NEW)
│   ├── Display Detected Data Types
│   │   ├── Numeric (int, float)
│   │   ├── Categorical (object/string)
│   │   ├── Boolean
│   │   └── DateTime
│   │
│   ├── Manual Data Type Conversion (Typecasting)
│   │   ├── Convert to Numeric (int/float)
│   │   ├── Convert to Categorical
│   │   ├── Convert to String
│   │   ├── Convert to Boolean
│   │   └── Convert to DateTime
│   │
│   ├── Categorical Variable Encoding
│   │   ├── Label Encoding (0,1,2...)
│   │   ├── One-Hot Encoding (Dummy Variables)
│   │   ├── Ordinal Encoding (user-defined order)
│   │   └── View Mapping (e.g., Male=0, Female=1)
│   │
│   ├── Column Renaming Option
│   ├── Preview Before/After Changes
│   └── Apply Transformations
│
├── EDA Module (Exploratory Data Analysis)
│   ├── Missing Values Analysis
│   │   ├── Heatmap of missing values (Plotly)
│   │   ├── Count and percentage per column
│   │   └── Imputation Options:
│   │       ├── Mean (for numeric)
│   │       ├── Median (for numeric)
│   │       ├── Mode (for all types)
│   │       ├── Forward Fill / Backward Fill
│   │       ├── Drop rows/columns
│   │       └── Custom value
│   │
│   ├── Data Types Overview
│   ├── Duplicate Rows Detection and Removal
│   ├── Outlier Detection (IQR method, Z-score)
│   │   └── Options: Keep / Remove / Cap (Winsorize)
│   └── Download Cleaned Dataset (CSV / XLSX)
│
├── Descriptive Statistics Module
│   ├── Full Summary Table (count, mean, std, min, 25%, 50%, 75%, max)
│   ├── Skewness and Kurtosis
│   ├── Variance and Standard Deviation
│   ├── Frequency Tables (for Categorical Variables)
│   ├── Correlation Matrix (Pearson, Spearman, Kendall)
│   │   └── Interactive heatmap with color coding
│   └── Export Summary to Excel
│
├── Visualization Module (SPSS-like Plot Builder)
│   ├── X-Axis Variable Selector
│   ├── Y-Axis Variable Selector
│   ├── Color / Label / Group-by Variable (3rd axis / class)
│   ├── Plot Size Adjuster (Width x Height in pixels)
│   ├── Export Format: PNG / SVG / JPEG
│   ├── DPI Selector: 72 / 150 / 300 dpi
│   │
│   └── Chart Types:
│       ├── Bar Chart (Simple, Grouped, Stacked)
│       ├── Line Chart
│       ├── Scatter Plot (with optional trendline)
│       ├── Box Plot
│       ├── Violin Plot
│       ├── Histogram (adjustable bins)
│       ├── Pie Chart / Donut Chart
│       ├── Heatmap (Correlation / Custom)
│       ├── Area Chart
│       ├── Bubble Chart (X, Y, Size = 3 variables)
│       ├── Pair Plot / Scatterplot Matrix
│       ├── QQ Plot (Normality check)
│       └── Count Plot / Frequency Plot
│
├── Statistical Tests Module
│   ├── Normality Tests (Shapiro-Wilk, KS Test, QQ Plot)
│   ├── Parametric Tests (t-tests, One-Way ANOVA)
│   ├── Non-Parametric Tests (Mann-Whitney, Kruskal-Wallis)
│   └── Chi-Square Test (independence + goodness of fit)
│
└── Modeling Module
    ├── Regression: Simple Linear, Multiple Linear, Logistic, Polynomial
    ├── Classification: KNN, Decision Tree, Random Forest, SVM, XGBoost
    ├── Clustering: K-Means (elbow plot + cluster visualization)
    └── Common: Train/Test split, feature selection, metrics, export predictions
```

---

## 5. 📱 Mobile-Specific Features

| Feature | Mobile Support |
|---|---|
| Dataset Upload (CSV/Excel) | Yes |
| EDA Summary | Yes |
| Missing Values Handling | Yes |
| Descriptive Statistics Table | Yes |
| Basic Plotting (Bar, Histogram) | Yes |
| Advanced Modeling | Simplified View |
| Plot Export | Yes |

---

## 6. 🎨 UI/UX Design Principles

- **Theme:** Dark professional (similar to IBM SPSS / RStudio dark theme)
- **Colors:** Navy blue (#0a1628), white, teal accent (#00d4aa)
- **Layout:** Sidebar navigation + main content area
- **Sections:** Clearly labeled tabs for each module
- **Feedback:** Loading spinners, success/error messages
- **Accessibility:** Readable fonts, contrast-compliant colors

---

## 7. 🚀 Deployment Strategy (Free Resources Only)

### Option 1 — Streamlit Community Cloud (RECOMMENDED)
- URL: https://share.streamlit.io
- Cost: 100% Free
- Steps: Push to GitHub → Connect Streamlit Cloud → Deploy
- Mobile: Accessible via any mobile browser

### Option 2 — Hugging Face Spaces
- URL: https://huggingface.co/spaces
- Cost: Free
- Good for: Sharing, backup deployment

### Option 3 — Render.com
- Cost: Free (sleeps after 15min inactivity)
- Good for: Always-on if upgraded

---

## 8. 📁 Project File Structure

```
statviz-pro/
│
├── app.py                        # Main Streamlit entry point
├── requirements.txt              # All Python dependencies
├── README.md                     # Documentation
├── .streamlit/
│   └── config.toml               # Theme + layout settings
│
├── modules/
│   ├── data_import.py            # File upload and preview
│   ├── datatype.py               # Data type conversion & encoding
│   ├── eda.py                    # EDA: missing values, outliers, duplicates
│   ├── descriptive.py            # Descriptive statistics
│   ├── visualization.py          # All Plotly chart functions
│   ├── statistical_tests.py      # Hypothesis tests
│   └── modeling.py               # ML models + evaluation
│
├── utils/
│   ├── helpers.py                # Utility functions
│   └── session_state.py          # Streamlit state management
│
└── assets/
    ├── logo.png
    └── sample_data/
        ├── iris.csv
        ├── titanic.csv
        └── heart_disease.csv
```

---

## 9. 📦 requirements.txt (All Free Libraries)

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
statsmodels>=0.14.0
openpyxl>=3.1.0
xlrd>=2.0.1
xgboost>=1.7.0
kaleido>=0.2.1
fpdf2>=2.7.0
```

---

## 10. 🗓️ Development Phases

### Phase 1 — MVP
- [x] PRD document
- [ ] App skeleton + navigation
- [ ] Data Import Module
- [ ] Data Type Management Module
- [ ] EDA Module (missing values, outliers)
- [ ] Descriptive Statistics Module
- [ ] Basic Visualization (5 chart types)
- [ ] Deploy to Streamlit Community Cloud

### Phase 2 — Full Features
- [ ] All 13 chart types with PNG/SVG/JPEG export
- [ ] Statistical Tests Module
- [ ] Regression + Classification Models
- [ ] Mobile-optimized layout
- [ ] Sample datasets

### Phase 3 — Advanced
- [ ] PDF Report Generation
- [ ] Clustering Module
- [ ] Time Series Analysis (ARIMA, NARX reference)

---

## 11. ✅ Acceptance Criteria

- Supports CSV, XLSX files up to 50MB (free-tier optimized)
- EDA completes in under 5 seconds for datasets up to 100k rows
- All plots are interactive (Plotly) and exportable (PNG/SVG/JPEG)
- App loads on mobile browsers
- Deployment is public and free
- Missing value imputation works for all methods (mean, median, mode, fill, drop)
- Data type conversion works correctly
- Categorical encoding works (Label + One-Hot)
- Export dataset to CSV and XLSX
- At least 5 regression/classification models with accuracy metrics

---

*PRD Version: 2.0 | Author: Usama | Date: March 2026 | Status: Ready for Development*

