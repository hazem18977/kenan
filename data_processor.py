"""
Data processing module for kinetic modeling analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import csv
from io import StringIO


def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the data has the required structure.
    А/А0 column is optional and will be calculated if missing.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['т, мин', 'А', 'А0']
    optional_columns = ['А/А0']

    if df.empty:
        return False, "Файл пуст"

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        return False, f"Отсутствуют обязательные столбцы: {', '.join(missing_required)}"

    return True, ""


def convert_european_decimal(value):
    """
    Convert European decimal format (comma as decimal separator) to float.

    Args:
        value: Value to convert (can be string, float, or int)

    Returns:
        Float value or NaN if conversion fails
    """
    if pd.isna(value):
        return np.nan

    # If already numeric, return as is
    if isinstance(value, (int, float)):
        return float(value)

    # Convert to string and handle European format
    str_value = str(value).strip()

    # Replace comma with dot for decimal separator
    str_value = str_value.replace(',', '.')

    try:
        return float(str_value)
    except (ValueError, TypeError):
        return np.nan


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the kinetic data by cleaning and calculating derived columns.
    Automatically calculates А/А0 if missing.
    Enhanced to handle European decimal formats.

    Args:
        df: Raw DataFrame from file or manual entry

    Returns:
        Processed DataFrame with additional columns
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Convert required columns to numeric, handling European decimal format
    for col in ['т, мин', 'А', 'А0']:
        if col in processed_df.columns:
            # First try standard pandas conversion
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            # If we have NaN values, try European decimal conversion
            nan_mask = processed_df[col].isna()
            if nan_mask.any():
                # Apply European decimal conversion to NaN values
                original_values = df[col][nan_mask]
                converted_values = original_values.apply(convert_european_decimal)
                processed_df.loc[nan_mask, col] = converted_values

    # Calculate А/А0 if missing or convert to numeric if present
    if 'А/А0' not in processed_df.columns:
        processed_df['А/А0'] = processed_df['А'] / processed_df['А0']
    else:
        # First try standard pandas conversion
        processed_df['А/А0'] = pd.to_numeric(processed_df['А/А0'], errors='coerce')

        # If we have NaN values, try European decimal conversion
        nan_mask = processed_df['А/А0'].isna()
        if nan_mask.any():
            original_values = df['А/А0'][nan_mask]
            converted_values = original_values.apply(convert_european_decimal)
            processed_df.loc[nan_mask, 'А/А0'] = converted_values

        # Recalculate if still NaN after conversion
        nan_mask = processed_df['А/А0'].isna()
        if nan_mask.any():
            processed_df.loc[nan_mask, 'А/А0'] = processed_df.loc[nan_mask, 'А'] / processed_df.loc[nan_mask, 'А0']

    # Drop rows with NaN values in essential columns
    processed_df.dropna(subset=['т, мин', 'А', 'А0', 'А/А0'], inplace=True)

    # Filter out non-positive values in 'А/А0' for logarithm calculation
    if len(processed_df) > 0 and (processed_df['А/А0'] <= 0).any():
        processed_df = processed_df[processed_df['А/А0'] > 0].copy()

    # Filter out zero or negative values in 'А' for inverse calculation
    if len(processed_df) > 0 and (processed_df['А'] <= 0).any():
        processed_df = processed_df[processed_df['А'] > 0].copy()

    # Filter out zero or negative values in 'А0'
    if len(processed_df) > 0 and (processed_df['А0'] <= 0).any():
        processed_df = processed_df[processed_df['А0'] > 0].copy()

    # Calculate derived columns only if we have data
    if len(processed_df) > 0:
        processed_df['ln_A_A0'] = np.log(processed_df['А/А0'])
        processed_df['inv_A'] = 1 / processed_df['А']

    return processed_df


def detect_csv_delimiter(file_content: str) -> str:
    """
    Detect the delimiter used in a CSV file.

    Args:
        file_content: String content of the CSV file

    Returns:
        Detected delimiter (comma, semicolon, or tab)
    """
    # Try to detect delimiter using csv.Sniffer
    try:
        sample = file_content[:1024]  # Use first 1KB for detection
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample, delimiters=',;\t').delimiter
        return delimiter
    except:
        # Fallback: count occurrences of common delimiters
        delimiters = [',', ';', '\t']
        delimiter_counts = {}

        for delim in delimiters:
            delimiter_counts[delim] = file_content.count(delim)

        # Return the delimiter with the highest count
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        return best_delimiter if delimiter_counts[best_delimiter] > 0 else ','


def read_csv_file(uploaded_file) -> pd.DataFrame:
    """
    Read CSV file with automatic delimiter detection and encoding handling.
    Enhanced to handle various encodings and European CSV formats.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        DataFrame with CSV data
    """
    # Extended list of encodings to try, prioritizing common ones
    encodings = [
        'utf-8', 'utf-8-sig',  # UTF-8 with and without BOM
        'windows-1251',        # Cyrillic (Russian, Bulgarian, etc.)
        'windows-1252',        # Western European
        'cp1252',             # Windows Western European
        'iso-8859-1',         # Latin-1
        'iso-8859-15',        # Latin-9 (includes Euro symbol)
        'latin1',             # Latin-1 alias
        'cp850',              # DOS Latin-1
        'cp1250',             # Central European
        'utf-16',             # UTF-16
        'utf-16le',           # UTF-16 Little Endian
        'utf-16be'            # UTF-16 Big Endian
    ]

    content = None
    used_encoding = None
    last_error = None

    # Try different encodings
    for encoding in encodings:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            content = uploaded_file.read().decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    if content is None:
        error_msg = f"Не удалось определить кодировку файла. Последняя ошибка: {str(last_error)}"
        error_msg += "\nПопробуйте сохранить файл в UTF-8 или обратитесь за помощью."
        raise ValueError(error_msg)

    # Detect delimiter
    delimiter = detect_csv_delimiter(content)

    # Try reading CSV with different configurations
    read_attempts = [
        # Standard format
        {'sep': delimiter},
        # European format (comma as decimal separator)
        {'sep': delimiter, 'decimal': ','},
        # European format with different thousands separator
        {'sep': delimiter, 'decimal': ',', 'thousands': '.'},
        # Try with different quoting
        {'sep': delimiter, 'quotechar': '"'},
        {'sep': delimiter, 'decimal': ',', 'quotechar': '"'},
        # Try with skipinitialspace
        {'sep': delimiter, 'skipinitialspace': True},
        {'sep': delimiter, 'decimal': ',', 'skipinitialspace': True}
    ]

    df = None
    last_read_error = None

    for attempt_params in read_attempts:
        try:
            df = pd.read_csv(StringIO(content), **attempt_params)
            # Check if we got valid data
            if not df.empty and len(df.columns) > 1:
                break
        except Exception as e:
            last_read_error = e
            continue

    if df is None or df.empty:
        error_msg = f"Ошибка чтения CSV файла. Последняя ошибка: {str(last_read_error)}"
        error_msg += f"\nИспользованная кодировка: {used_encoding}"
        error_msg += f"\nОбнаруженный разделитель: '{delimiter}'"
        error_msg += "\nУбедитесь, что файл содержит корректные данные в формате CSV."
        raise ValueError(error_msg)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the processed data.

    Args:
        df: Processed DataFrame

    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_points': len(df),
        'time_range': (df['т, мин'].min(), df['т, мин'].max()),
        'a_range': (df['А'].min(), df['А'].max()),
        'a0_value': df['А0'].iloc[0] if len(df) > 0 else None,
        'a_a0_range': (df['А/А0'].min(), df['А/А0'].max())
    }
