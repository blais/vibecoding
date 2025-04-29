#!/usr/bin/env python3
"""A Python computer program that will read the tax-rates.csv file attached
which contains marginal taxation rates by income tranches and, given a filing
status and a income amount will return the cumulative amount of taxes paid for
that amount, including the dollar amount and the percentage.
"""

import argparse
import io
import sys
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np


def load_and_clean_data(
    csv_path: str = "tax-rates.csv",
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[str, str]]]:
    """Loads tax rate data from CSV, cleans it, and returns brackets and deductions."""
    try:
        df_tax_rates: pd.DataFrame = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        sys.exit(1)

    # Function to clean currency values: remove '$', ',', handle NaN/None/empty strings
    def clean_currency(value: Any) -> float:
        if pd.isna(value) or value == "":
            return np.nan  # Represent missing numeric values as NaN
        try:
            # Remove '$' and ',' and convert to float
            return float(str(value).replace("$", "").replace(",", ""))
        except ValueError:
            return np.nan  # Handle any conversion errors

    # Function to clean percentage values
    def clean_percentage(value: Any) -> float:
        if pd.isna(value) or value == "":
            return np.nan
        try:
            # Remove '%' and convert to decimal
            return float(str(value).replace("%", "")) / 100.0
        except ValueError:
            return np.nan

    # Extract standard deductions from the first row
    standard_deductions_raw = df_tax_rates.iloc[0]
    standard_deductions = {
        "Single": clean_currency(standard_deductions_raw["Single Min"]),
        "Married filing jointly": clean_currency(
            standard_deductions_raw["Married filing jointly Min"]
        ),
        "Married filing separately": clean_currency(
            standard_deductions_raw["Married filing separately Min"]
        ),
        "Head of household": clean_currency(
            standard_deductions_raw["Head of household Min"]
        ),
    }

    # Remove the standard deduction row and use the next row for processing tax brackets
    df_brackets = df_tax_rates[1:].reset_index(drop=True)

    # Clean the 'Tax Rate' column
    df_brackets["Rate"] = df_brackets["Tax Rate"].apply(clean_percentage)

    # Define column mappings for different filing statuses
    status_columns = {
        "Single": ("Single Min", "Single Max"),
        "Married filing jointly": (
            "Married filing jointly Min",
            "Married filing jointly Max",
        ),
        "Married filing separately": (
            "Married filing separately Min",
            "Married filing separately Max",
        ),
        "Head of household": ("Head of household Min", "Head of household Max"),
    }

    # Clean Min/Max columns for all statuses
    for status, (min_col, max_col) in status_columns.items():
        df_brackets[f"{status}_Min"] = df_brackets[min_col].apply(clean_currency)
        df_brackets[f"{status}_Max"] = df_brackets[max_col].apply(clean_currency)
        # Replace NaN in Max column (top bracket) with infinity
        df_brackets[f"{status}_Max"] = df_brackets[f"{status}_Max"].fillna(np.inf)

    # Select only the necessary cleaned columns for calculations
    cols_to_keep = ["Rate"] + [
        f"{s}_{m}" for s in status_columns for m in ["Min", "Max"]
    ]
    df_final_brackets = df_brackets[cols_to_keep].copy()

    return df_final_brackets, standard_deductions, status_columns


# --- Tax Calculation Function ---


def calculate_tax(
    gross_income: float,
    filing_status: str,
    brackets_df: pd.DataFrame,
    deductions: Dict[str, float],
    status_columns: Dict[str, Tuple[str, str]],
) -> Tuple[float, float, float, float]:
    """Calculates the total tax and effective tax rate."""

    if filing_status not in deductions:
        raise ValueError(
            f"Invalid filing status provided: {filing_status}. Valid options are: {list(deductions.keys())}"
        )

    # Get the standard deduction for the status
    deduction = deductions.get(filing_status, 0)
    if pd.isna(deduction):  # Handle if deduction wasn't properly extracted
        print(
            f"Warning: Standard deduction not found or invalid for {filing_status}. Using $0."
        )
        deduction = 0

    # Calculate taxable income (cannot be less than 0)
    taxable_income = max(0, gross_income - deduction)

    # Get the relevant Min/Max column names for the status
    min_col_name = f"{filing_status}_Min"
    max_col_name = f"{filing_status}_Max"

    total_tax = 0.0
    last_bracket_max = 0.0

    # Iterate through each tax bracket, ensuring they are sorted by minimum income
    sorted_brackets = brackets_df.sort_values(by=min_col_name)

    for index, row in sorted_brackets.iterrows():
        rate = row["Rate"]
        bracket_min = row[min_col_name]
        bracket_max = row[max_col_name]

        # Use the previous bracket's max as the effective minimum for calculation
        # This handles potential overlaps or gaps if data isn't perfectly contiguous
        # However, for standard marginal brackets, bracket_min should align with the previous max
        effective_bracket_min = last_bracket_max

        # Ensure rate is valid
        if pd.isna(rate):
            continue  # Skip rows where rate is missing

        # If taxable income is below this bracket's minimum, we might be done or skip
        if taxable_income <= effective_bracket_min:
            continue  # Income already taxed by lower brackets

        # Calculate the amount of income that falls *within* this specific bracket
        # Income subject to this rate = min(taxable_income, bracket_max) - effective_bracket_min
        taxable_amount_in_bracket = max(
            0, min(taxable_income, bracket_max) - effective_bracket_min
        )

        # Calculate the tax for this portion of income
        tax_for_bracket = taxable_amount_in_bracket * rate

        # Add to the total tax
        total_tax += tax_for_bracket

        # Update the max income taxed so far for the next iteration
        last_bracket_max = bracket_max

        # If taxable income doesn't exceed this bracket's max, we're done taxing
        if taxable_income <= bracket_max:
            break  # No need to check higher brackets

    # Calculate effective tax rate (as a percentage of gross income)
    effective_rate = (total_tax / gross_income) * 100 if gross_income > 0 else 0

    return total_tax, effective_rate, taxable_income, deduction


# --- Constants ---
FILING_STATUS_ACRONYMS = {
    "S": "Single",
    "MFJ": "Married filing jointly",
    "MFS": "Married filing separately",
    "HOH": "Head of household",
}


# --- Main Program Execution ---


def main() -> None:
    """Parses command-line arguments and calculates taxes."""
    # Load and clean data first
    df_final_brackets: pd.DataFrame
    standard_deductions: Dict[str, float]
    status_columns: Dict[str, Tuple[str, str]]
    df_final_brackets, standard_deductions, status_columns = load_and_clean_data()

    parser = argparse.ArgumentParser(
        description="Calculate US federal income tax based on filing status and gross income."
    )
    # Combine full names and acronyms for choices and help message
    all_status_options = list(standard_deductions.keys()) + list(
        FILING_STATUS_ACRONYMS.keys()
    )
    status_help = f"Your filing status. Options: {', '.join(standard_deductions.keys())}. Acronyms: {', '.join(FILING_STATUS_ACRONYMS.keys())}."

    parser.add_argument(
        "-s",
        "--filing-status",
        required=True,
        choices=all_status_options,
        help=status_help,
        metavar="STATUS", # Use a generic metavar
    )
    parser.add_argument(
        "-i",
        "--gross-income",
        required=True,
        type=float,
        help="Your total gross income.",
    )

    args = parser.parse_args()

    # Resolve acronym if provided
    chosen_filing_status = FILING_STATUS_ACRONYMS.get(
        args.filing_status, args.filing_status
    )
    chosen_gross_income = args.gross_income

    if chosen_gross_income < 0:
        print("Error: Gross income cannot be negative.")
        sys.exit(1)

    # Calculate and display results
    try:
        total_tax_owed, eff_rate, taxable_inc, std_deduction = calculate_tax(
            chosen_gross_income,
            chosen_filing_status,
            df_final_brackets,
            standard_deductions,
            status_columns,  # Pass status_columns here
        )

        print("\n--- Tax Calculation Results ---")
        print(f"Filing Status: {chosen_filing_status}")
        print(f"Gross Income: ${chosen_gross_income:,.2f}")
        print(f"Standard Deduction: ${std_deduction:,.2f}")
        print(f"Taxable Income: ${taxable_inc:,.2f}")
        print(f"\nTotal Tax Owed: ${total_tax_owed:,.2f}")
        print(f"Effective Tax Rate: {eff_rate:.2f}%")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during calculation: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
        sys.exit(1)


if __name__ == "__main__":
    main()
