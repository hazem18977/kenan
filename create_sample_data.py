"""
Script to create sample Excel data for testing the kinetic modeling application.
"""

import pandas as pd
import numpy as np

# Create sample kinetic data
np.random.seed(42)  # For reproducible results

# Time points (minutes)
time = np.array([0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])

# Simulate PFO kinetics with some noise
k1_true = 0.05  # True rate constant
A0 = 100  # Initial concentration

# Generate true PFO data
A_A0_true = np.exp(-k1_true * time)
A_true = A0 * A_A0_true

# Add some realistic noise
noise_level = 0.02
A_measured = A_true * (1 + np.random.normal(0, noise_level, len(time)))
A_A0_measured = A_measured / A0

# Ensure no negative values
A_measured = np.maximum(A_measured, 0.1)
A_A0_measured = np.maximum(A_A0_measured, 0.001)

# Create DataFrame
sample_data = pd.DataFrame({
    'т, мин': time,
    'А': A_measured,
    'А0': np.full(len(time), A0),
    'А/А0': A_A0_measured
})

# Round values for cleaner presentation
sample_data['А'] = sample_data['А'].round(2)
sample_data['А/А0'] = sample_data['А/А0'].round(4)

# Create multiple sheets with different conditions
sheets_data = {}

# Sheet 1: pH 10 (faster reaction)
k1_ph10 = 0.08
A_true_ph10 = A0 * np.exp(-k1_ph10 * time)
A_measured_ph10 = A_true_ph10 * (1 + np.random.normal(0, noise_level, len(time)))
A_measured_ph10 = np.maximum(A_measured_ph10, 0.1)
A_A0_measured_ph10 = A_measured_ph10 / A0

sheets_data['pH 10'] = pd.DataFrame({
    'т, мин': time,
    'А': A_measured_ph10.round(2),
    'А0': np.full(len(time), A0),
    'А/А0': A_A0_measured_ph10.round(4)
})

# Sheet 2: pH 3 (slower reaction)
k1_ph3 = 0.02
A_true_ph3 = A0 * np.exp(-k1_ph3 * time)
A_measured_ph3 = A_true_ph3 * (1 + np.random.normal(0, noise_level, len(time)))
A_measured_ph3 = np.maximum(A_measured_ph3, 0.1)
A_A0_measured_ph3 = A_measured_ph3 / A0

sheets_data['pH 3'] = pd.DataFrame({
    'т, мин': time,
    'А': A_measured_ph3.round(2),
    'А0': np.full(len(time), A0),
    'А/А0': A_A0_measured_ph3.round(4)
})

# Sheet 3: pH 8 (medium reaction)
sheets_data['pH 8'] = sample_data.copy()

# Sheet 4: Empty sheet for testing error handling
sheets_data['Пустой лист'] = pd.DataFrame()

# Save to Excel with multiple sheets
with pd.ExcelWriter('sample_kinetic_data_multi_sheet.xlsx', engine='openpyxl') as writer:
    for sheet_name, data in sheets_data.items():
        data.to_excel(writer, sheet_name=sheet_name, index=False)

# Also save single sheet version
sample_data.to_excel('sample_kinetic_data.xlsx', index=False)

print("Sample data created successfully!")
print("\nMulti-sheet file created: sample_kinetic_data_multi_sheet.xlsx")
print("Available sheets:")
for sheet_name in sheets_data.keys():
    if not sheets_data[sheet_name].empty:
        print(f"  - {sheet_name}: {len(sheets_data[sheet_name])} rows")
    else:
        print(f"  - {sheet_name}: empty sheet")

print("\nSingle sheet file: sample_kinetic_data.xlsx")
print("\nSample data preview (pH 8):")
print(sample_data.head(10))

print(f"\nData characteristics:")
print(f"Time range: {time[0]} - {time[-1]} minutes")
print(f"Initial concentration: {A0}")
print(f"pH 10 - Final A/A0 ratio: {A_A0_measured_ph10[-1]:.4f} (k1≈{k1_ph10})")
print(f"pH 8 - Final A/A0 ratio: {A_A0_measured[-1]:.4f} (k1≈{k1_true})")
print(f"pH 3 - Final A/A0 ratio: {A_A0_measured_ph3[-1]:.4f} (k1≈{k1_ph3})")
