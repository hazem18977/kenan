# Kinetic Modeling Analysis Web Application
# Веб-приложение для анализа кинетического моделирования

A Streamlit web application for analyzing kinetic data using Pseudo-First Order (PFO) and Pseudo-Second Order (PSO) models.

## Features / Возможности

- **Multiple Input Methods**: Upload Excel/CSV files OR enter data manually through web interface
- **File Format Support**: Excel (.xlsx, .xls) and CSV (.csv) with automatic delimiter detection
- **Multi-Sheet Support**: Select and analyze specific worksheets from Excel files with multiple sheets
- **Manual Data Entry**: Clean table editor without pre-filled templates
- **Auto-calculation**: Automatic calculation of A/A0 ratios when missing from input data
- **Professional Interface**: Clean, emoji-free design suitable for business/scientific use
- **Automatic Processing**: Data validation and preprocessing for all input methods
- **Stable Point Detection**: Automatic selection of linear regions (fixed threshold 0.1)
- **Model Fitting**: PFO and PSO kinetic models with curve fitting
- **Performance Metrics**: MAPE and R² calculations
- **Static Visualizations**: High-quality Matplotlib plots
- **Russian Interface**: Complete Russian language interface
- **Results Export**: Download results as Excel files

## Installation / Установка

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## File Structure / هيكل الملفات

```
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data preprocessing functions
├── kinetic_models.py      # PFO and PSO modeling functions
├── visualization.py       # Plotting and visualization functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Required Data Format / Требуемый формат данных

Your Excel or CSV file must contain the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `т, мин` | Time in minutes | Yes |
| `А` | Concentration A | Yes |
| `А0` | Initial concentration | Yes |
| `А/А0` | Ratio A/A0 | No (calculated automatically if missing) |

### Example Data / Пример данных

| т, мин | А   | А0  | А/А0 |
|--------|-----|-----|------|
| 0      | 100 | 100 | 1.0  |
| 5      | 85  | 100 | 0.85 |
| 10     | 72  | 100 | 0.72 |
| 15     | 61  | 100 | 0.61 |
| 20     | 52  | 100 | 0.52 |

## Models / Модели

### Pseudo-First Order (PFO)
- **Equation**: ln(A/A₀) = -k₁ × t
- **Linear form**: ln(A/A₀) vs time
- **Parameter**: k₁ (rate constant, min⁻¹)

### Pseudo-Second Order (PSO)
- **Equation**: 1/A = 1/A₀ + k₂ × t
- **Linear form**: 1/A vs time
- **Parameter**: k₂ (rate constant, L/(mg·min))

## Features Details / Подробности возможностей

### Automatic Stable Point Detection
The application automatically identifies the linear region of your data using slope analysis with a fixed threshold of 0.1 for consistent results.

### Model Performance Metrics
- **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy
- **R² (Coefficient of Determination)**: Measures goodness of fit

### Static Visualizations
- **Main Plots**: Show experimental data and model fits using Matplotlib
- **Professional Quality**: High-resolution plots suitable for publications
- **Downloadable Results**: Export all results to Excel

## Usage Tips / Советы по использованию

1. **Input Methods**: Choose between file upload (Excel/CSV) or manual data entry based on your needs
2. **File Formats**: Upload Excel (.xlsx, .xls) or CSV files - CSV delimiter is detected automatically
3. **Multi-Sheet Files**: When uploading Excel files with multiple sheets, select the appropriate sheet from the dropdown
4. **Manual Entry**: Use the clean table editor to input data from scratch - A/A0 ratios are calculated automatically
5. **Missing A/A0**: If your file doesn't contain A/A0 column, it will be calculated automatically from A and A0
6. **Data Quality**: Ensure your data doesn't contain missing values or zeros in concentration columns
7. **Automatic Selection**: The application uses a fixed threshold (0.1) for consistent stable point selection
8. **Model Comparison**: Compare R² and MAPE values to choose the best model
9. **Visual Inspection**: Always check the plots to validate model fits

## Troubleshooting / Устранение неполадок

### Common Issues:
1. **"Missing columns" error**: Check that your file has the required column names (т, мин; А; А0)
2. **"No valid data" error**: Ensure your data doesn't contain negative values or zeros
3. **"Empty sheet" error**: Select a different sheet if working with multi-sheet Excel files
4. **CSV delimiter issues**: The app auto-detects delimiters, but ensure your CSV is properly formatted
5. **Manual entry validation errors**: Ensure time values are in ascending order and all concentrations are positive
6. **Poor model fit**: Check your data quality and ensure it follows kinetic behavior

### Data Requirements:
- Time values must be positive and increasing
- Concentration values must be positive
- A/A₀ ratios must be between 0 and 1 (typically)

## Extension Ideas / أفكار للتطوير

The modular structure makes it easy to extend the application:

- Add more kinetic models (Zero-order, Elovich, etc.)
- Implement batch processing for multiple files
- Add statistical analysis and confidence intervals
- Include temperature-dependent modeling
- Add data export in different formats

## Dependencies / التبعيات

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- openpyxl >= 3.1.0


## License / الترخيص

This project is open source and available under the MIT License.
