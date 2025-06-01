"""
Streamlit Web Application for Kinetic Modeling Analysis
Веб-приложение для анализа кинетического моделирования
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Import custom modules
from data_processor import validate_data_structure, preprocess_data, get_data_summary, read_csv_file
from kinetic_models import (
    find_stable_points, fit_pfo_model, fit_pso_model,
    create_results_summary, create_detailed_results
)
from visualization import create_matplotlib_plots

# Configure page
st.set_page_config(
    page_title="Анализ кинетического моделирования",
    page_icon="🧑‍🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Анализ кинетического моделирования")

    # Student and supervisor information
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <p style="margin: 0; font-size: 14px;">
            <strong>СТУДЕНТ:</strong> Алсади К. <br>
            <strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Параметры")

        # Using Matplotlib for all plots
        plot_type = "Matplotlib"

        st.markdown("---")
        st.markdown("### Требования к файлу")
        st.markdown("""
        **Обязательные столбцы:**
        - `т, мин` (Время в минутах)
        - `А` (Концентрация A)

        **Опциональные столбцы:**
        - `А0` (Начальная концентрация) - если отсутствует, используется первое значение А
        - `А/А0` (Отношение A/A0) - рассчитывается автоматически если отсутствует

        **Поддерживаемые форматы:**
        - Excel (.xlsx, .xls)
        - CSV (.csv) с автоопределением разделителя
        """)

    # Data input method selection
    st.header("Ввод данных")

    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["Загрузить файл", "Ввести данные вручную"],
        index=0,
        horizontal=True
    )

    df = None
    selected_sheet = None

    if input_method == "Загрузить файл":
        # File upload section
        st.subheader("Загрузка файла")
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['xlsx', 'xls', 'csv'],
            help="Загрузите файл Excel или CSV с кинетическими данными"
        )

        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    # Handle CSV files
                    df = read_csv_file(uploaded_file)

                    # Check if file is empty
                    if df.empty:
                        st.error("Файл CSV пуст")
                        return

                else:
                    # Handle Excel files
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names

                    # Add sheet selector to sidebar if multiple sheets exist
                    selected_sheet = None
                    if len(sheet_names) > 1:
                        with st.sidebar:
                            st.markdown("---")
                            selected_sheet = st.selectbox(
                                "Выберите лист Excel",
                                sheet_names,
                                index=0,
                                help="Выберите лист для анализа из доступных в файле"
                            )
                    else:
                        selected_sheet = sheet_names[0]

                    # Read the selected sheet
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                        # Check if sheet is empty
                        if df.empty:
                            st.error(f"Лист '{selected_sheet}' пуст")
                            if len(sheet_names) > 1:
                                st.error("Попробуйте выбрать другой лист.")
                            return

                    except Exception as sheet_error:
                        st.error(f"Ошибка чтения листа '{selected_sheet}': {str(sheet_error)}")
                        if len(sheet_names) > 1:
                            st.error("Попробуйте выбрать другой лист.")
                        return

                # Validate structure (common for both CSV and Excel)
                is_valid, error_message = validate_data_structure(df)

                if not is_valid:
                    st.error(f"{error_message}")
                    if file_extension != 'csv' and len(sheet_names) > 1:
                        st.error(f"Лист '{selected_sheet}' не содержит требуемых данных. Попробуйте выбрать другой лист.")
                    return

                st.success("Файл успешно загружен!")
                if file_extension != 'csv' and len(sheet_names) > 1:
                    st.info(f"Анализируется лист: **{selected_sheet}**")

                # Show raw data preview
                with st.expander("Предварительный просмотр данных"):
                    st.dataframe(df, use_container_width=True)
                    st.info(f"Общее количество строк: {len(df)}")

                    # Debug information for CSV files
                    if file_extension == 'csv':
                        st.markdown("**Информация о типах данных:**")
                        for col in ['т, мин', 'А', 'А0', 'А/А0']:
                            if col in df.columns:
                                sample_values = df[col].head(3).tolist()
                                data_type = df[col].dtype
                                st.text(f"{col}: {data_type} | Примеры: {sample_values}")

                # Data editing section for uploaded files
                st.subheader("Редактирование данных")
                st.info("Вы можете отредактировать загруженные данные перед анализом. Столбец А/А0 будет пересчитан автоматически.")

                # Prepare data for editing
                edit_df = df.copy()

                # Configure column settings for the data editor
                column_config = {
                    "т, мин": st.column_config.NumberColumn(
                        "Время (мин)",
                        help="Время в минутах",
                        min_value=0.0,
                        step=0.0001,
                        format="%.4f"
                    ),
                    "А": st.column_config.NumberColumn(
                        "Концентрация А",
                        help="Концентрация А",
                        min_value=0.0,
                        step=0.0001,
                        format="%.5f"
                    ),
                    "А0": st.column_config.NumberColumn(
                        "Начальная концентрация А0",
                        help="Начальная концентрация А0",
                        min_value=0.0,
                        step=0.0001,
                        format="%.5f"
                    )
                }

                # Add А/А0 column configuration if it exists
                if 'А/А0' in edit_df.columns:
                    column_config["А/А0"] = st.column_config.NumberColumn(
                        "Отношение А/А0",
                        help="Отношение А/А0 (будет пересчитано автоматически)",
                        min_value=0.0,
                        step=0.0001,
                        format="%.4f",
                        disabled=True  # Make this column read-only since it's calculated
                    )

                # Create editable data table
                edited_df = st.data_editor(
                    edit_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config=column_config,
                    key="uploaded_data_editor"
                )

                # Auto-calculate А/А0 if А and А0 columns exist
                if 'А' in edited_df.columns and 'А0' in edited_df.columns:
                    # Create a copy to avoid modifying the original
                    processed_edited_df = edited_df.copy()

                    # Remove rows with invalid data
                    valid_mask = (processed_edited_df['А'] > 0) & (processed_edited_df['А0'] > 0) & (processed_edited_df['т, мин'] >= 0)
                    processed_edited_df = processed_edited_df[valid_mask]

                    if not processed_edited_df.empty:
                        # Recalculate А/А0
                        processed_edited_df['А/А0'] = processed_edited_df['А'] / processed_edited_df['А0']
                        processed_edited_df['А/А0'] = processed_edited_df['А/А0'].round(4)

                        # Update df to use the edited and processed data
                        df = processed_edited_df.copy()

                        # Show updated preview
                        with st.expander("Предварительный просмотр отредактированных данных"):
                            st.dataframe(df, use_container_width=True)
                            st.info(f"Действительных строк после редактирования: {len(df)}")
                    else:
                        st.warning("Нет действительных данных после редактирования. Убедитесь, что все значения положительные.")
                        df = None

            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")

        else:
            # Instructions when no file is uploaded
            st.info("Пожалуйста, загрузите файл для начала анализа")

    else:  # Manual data entry
        st.subheader("Ручной ввод данных")

        # Initialize session state for tracking first A value
        if 'first_a_value' not in st.session_state:
            st.session_state.first_a_value = None
        if 'manual_data_initialized' not in st.session_state:
            st.session_state.manual_data_initialized = False

        # Create empty template data
        default_data = pd.DataFrame({
            'т, мин': [0.0],
            'А': [0.0]
        })

        st.info("Введите данные в таблицу ниже. Столбец А/А0 будет рассчитан автоматически.")

        # Data editor
        edited_data = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "т, мин": st.column_config.NumberColumn(
                    "Время (мин)",
                    help="Время в минутах",
                    min_value=0.0,
                    step=0.0001,
                    format="%.4f"
                ),
                "А": st.column_config.NumberColumn(
                    "Концентрация А",
                    help="Концентрация А",
                    min_value=0.0,
                    step=0.0001,
                    format="%.5f"
                ),
                "А0": st.column_config.NumberColumn(
                    "Начальная концентрация А0",
                    help="Начальная концентрация А0",
                    min_value=0.0,
                    step=0.0001,
                    format="%.5f"
                )
            }
        )

        # Auto-calculate A/A0 ratios and show preview
        if not edited_data.empty and 'А' in edited_data.columns and 'А0' in edited_data.columns:
            # Remove rows with zero or negative values
            valid_data = edited_data[(edited_data['А'] > 0) & (edited_data['А0'] > 0) & (edited_data['т, мин'] >= 0)]

                if not valid_data.empty:
                    # Calculate A/A0 ratios
                    valid_data = valid_data.copy()
                    valid_data['А/А0'] = valid_data['А'] / valid_data['А0']
                    valid_data['А/А0'] = valid_data['А/А0'].round(4)

                    # Show preview with calculated ratios (only visible columns plus А/А0)
                    st.subheader("Предварительный просмотр с рассчитанными А/А0")
                    display_data = valid_data[['т, мин', 'А', 'А/А0']].copy()
                    st.dataframe(display_data, use_container_width=True)
                    st.info(f"Автоопределение активно: А0 = {first_a_value:.5f} для всех расчетов")

                # Show auto-population info if applicable
                if len(processed_data) > 0 and processed_data.iloc[0]['А'] > 0:
                    st.success(f"Автозаполнение активно: А0 = {processed_data.iloc[0]['А']:.5f} для всех строк")

        # Validate manual data
        if st.button("Анализировать введенные данные", type="primary"):
            if edited_data.empty:
                st.error("Пожалуйста, введите данные для анализа")
            else:
                # Use the valid data with calculated A/A0
                if not edited_data.empty and 'А' in edited_data.columns and 'А0' in edited_data.columns:
                    # Remove rows with zero or negative values
                    valid_data = processed_data[(processed_data['А'] > 0) & (processed_data['А0'] > 0) & (processed_data['т, мин'] >= 0)]

                    if valid_data.empty:
                        st.error("Нет действительных данных. Убедитесь, что все значения положительные.")
                    else:
                        # Calculate A/A0 ratios
                        valid_data = valid_data.copy()
                        valid_data['А/А0'] = valid_data['А'] / valid_data['А0']

                        # Validate the processed data
                        is_valid, error_message = validate_data_structure(valid_data)

                        if not is_valid:
                            st.error(f"{error_message}")
                        else:
                            # Check for additional validation
                            if valid_data['т, мин'].duplicated().any():
                                st.error("Значения времени не должны повторяться")
                            elif not valid_data['т, мин'].is_monotonic_increasing:
                                st.error("Значения времени должны быть в возрастающем порядке")
                            else:
                                # Data is valid, use it for analysis
                                df = valid_data.copy()
                                st.success("Данные успешно введены!")
                                st.info(f"Введено {len(df)} точек данных")

    # Process data if we have valid data (from either source)
    if df is not None and not df.empty:
        pass  # Debug information removed as requested

        # Process data
        processed_df = preprocess_data(df)

        if len(processed_df) == 0:
            st.error("Нет действительных данных после обработки")
            st.error("Возможные причины:")
            st.error("- Все значения равны нулю или отрицательные")
            st.error("- Проблемы с форматом чисел (например, запятая вместо точки)")
            st.error("- Отсутствуют обязательные столбцы")
            return

        # Data summary
        summary = get_data_summary(processed_df)

        st.header("Сводка данных")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Действительные точки", summary['total_points'])
        with col2:
            st.metric("Диапазон времени", f"{summary['time_range'][0]:.1f} - {summary['time_range'][1]:.1f} мин")
        with col3:
            st.metric("Начальная концентрация", f"{summary['a0_value']:.3f}")
        with col4:
            st.metric("Диапазон A/A0", f"{summary['a_a0_range'][0]:.3f} - {summary['a_a0_range'][1]:.3f}")

        # Find stable points (using fixed threshold of 0.1)
        stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
        selected_data = processed_df.iloc[stable_indices].copy()

        st.header("Выбранные точки")
        st.info(f"Выбрано {len(selected_data)} точек из {len(processed_df)}")
        st.info(f"Выбранный временной диапазон: {selected_data['т, мин'].iloc[0]:.1f} - {selected_data['т, мин'].iloc[-1]:.1f} мин")

        # Fit models
        try:
            k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
            k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

            # Results summary
            st.header("Результаты моделирования")

            results_summary = create_results_summary(k1, k2, mape_pfo, mape_pso, r2_pfo, r2_pso)

            # Display summary table
            st.subheader("Сводка результатов")
            st.dataframe(results_summary, use_container_width=True)

            # Model comparison metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Модель PFO")
                st.metric("R² Score", f"{r2_pfo:.4f}")
                st.metric("MAPE (%)", f"{mape_pfo:.2f}")
                st.metric("Константа скорости k₁", f"{abs(k1):.5f} мин⁻¹")

            with col2:
                st.markdown("### Модель PSO")
                st.metric("R² Score", f"{r2_pso:.4f}")
                st.metric("MAPE (%)", f"{mape_pso:.2f}")
                st.metric("Константа скорости k₂", f"{k2:.5f} л/(мг·мин)")

            # Detailed results
            with st.expander("Подробные результаты"):
                detailed_results = create_detailed_results(pfo_predictions, pso_predictions)
                st.dataframe(detailed_results, use_container_width=True)

            # Plots
            st.header("Графики")

            # Generate Matplotlib plots
            fig_main = create_matplotlib_plots(processed_df, selected_data, pfo_predictions, pso_predictions, k1, k2)
            st.pyplot(fig_main)

            # Download results
            st.header("Скачать результаты")

            # Prepare download data
            download_data = {
                'Summary': results_summary,
                'Detailed_Results': detailed_results,
                'Selected_Data': selected_data,
                'PFO_Predictions': pfo_predictions,
                'PSO_Predictions': pso_predictions
            }

            # Create Excel file for download
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, data in download_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)

            st.download_button(
                label="Скачать результаты как файл Excel",
                data=output.getvalue(),
                file_name="kinetic_modeling_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Ошибка моделирования: {str(e)}")

    else:
        # Show instructions when no data is available
        if input_method == "Загрузить файл":
            pass
        else:
            st.info("Пожалуйста, введите данные и нажмите кнопку анализа")


if __name__ == "__main__":
    main()
