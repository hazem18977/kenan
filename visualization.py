"""
Visualization functions for kinetic modeling analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple


def create_matplotlib_plots(df: pd.DataFrame, selected_data: pd.DataFrame,
                           pfo_predictions: pd.DataFrame, pso_predictions: pd.DataFrame,
                           k1: float, k2: float) -> plt.Figure:
    """
    Create matplotlib plots for PFO and PSO models.

    Args:
        df: Full dataset
        selected_data: Selected stable points
        pfo_predictions: PFO model predictions
        pso_predictions: PSO model predictions
        k1: PFO rate constant
        k2: PSO rate constant

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PFO plot
    ax1.plot(df['т, мин'], df['ln_A_A0'], 'bo-', label='Экспериментальные данные (ln(A/A₀))', markersize=4)
    ax1.plot(pfo_predictions['т, мин'], pfo_predictions['PFO_pred_ln'], 'r--',
             label=f'Модель PFO (k₁={abs(k1):.5f})', linewidth=2)
    ax1.set_title('Модель псевдо-первого порядка')
    ax1.set_xlabel('Время, мин')
    ax1.set_ylabel('ln(A/A₀)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PSO plot
    ax2.plot(df['т, мин'], df['inv_A'], 'go-', label='Экспериментальные данные (1/A)', markersize=4)
    ax2.plot(pso_predictions['т, мин'], pso_predictions['PSO_pred_inv'], 'r--',
             label=f'Модель PSO (k₂={k2:.5f})', linewidth=2)
    ax2.set_title('Модель псевдо-второго порядка')
    ax2.set_xlabel('Время, мин')
    ax2.set_ylabel('1/A')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig