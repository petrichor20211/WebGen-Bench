import pandas as pd
import json
import os
from pathlib import Path
from scipy import stats  # Add for pearson correlation

def read_jsonl_data(jsonl_path):
    """
    Read data from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        list: List of dictionaries containing the data
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_accuracy(df):
    """
    Calculate accuracy by comparing webvoyager_res with manual evaluation results.
    Only Pass/Fail/Uncertain values are considered valid.
    
    Args:
        df: DataFrame containing the results
        
    Returns:
        float: Accuracy score
    """
    # Define valid values
    valid_values = ['Pass', 'Fail', 'Uncertain']
    
    # Filter rows where both columns contain valid values (case insensitive)
    valid_cases = df[
        df['webvoyager_res'].astype(str).str.strip().str.title().isin(valid_values) &
        df['人工评估结果'].astype(str).str.strip().str.title().isin(valid_values)
    ]
    
    if len(valid_cases) == 0:
        return 0.0
    
    # Count matches (case insensitive comparison)
    matches = valid_cases[
        valid_cases['webvoyager_res'].astype(str).str.strip().str.title() == 
        valid_cases['人工评估结果'].astype(str).str.strip().str.title()
    ]
    
    # Calculate accuracy
    accuracy = len(matches) / len(valid_cases) if len(valid_cases) > 0 else 0
    
    return accuracy

def create_statistics(df):
    """
    Calculate statistics for each website.
    
    Args:
        df: DataFrame containing the results
        
    Returns:
        DataFrame: Statistics for each website
    """
    stats = df.groupby('web_name').agg({
        'res': lambda x: f"{sum(x == 'YES')}/{len(x)} ({sum(x == 'YES')/len(x)*100:.1f}%)"
    }).reset_index()
    stats.columns = ['Website', 'Pass Rate']
    return stats

def clean_null_values(df, columns_to_clean):
    """
    Replace null/empty values with "Uncertain" in specified columns.
    
    Args:
        df: DataFrame to clean
        columns_to_clean: List of column names to check and clean
        
    Returns:
        DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    for col in columns_to_clean:
        if col in df_cleaned.columns:
            # Count null/empty values before cleaning
            null_count = df_cleaned[col].isna().sum()
            empty_count = (df_cleaned[col].astype(str).str.strip() == '').sum()
            nan_string_count = (df_cleaned[col].astype(str).str.strip().str.lower() == 'nan').sum()
            
            total_null_empty = null_count + empty_count + nan_string_count
            
            if total_null_empty > 0:
                print(f"Found {total_null_empty} null/empty values in column '{col}' (null: {null_count}, empty: {empty_count}, 'nan' strings: {nan_string_count})")
                
                # Replace null values, empty strings, and 'nan' strings with "Uncertain"
                mask = (df_cleaned[col].isna() | 
                       (df_cleaned[col].astype(str).str.strip() == '') |
                       (df_cleaned[col].astype(str).str.strip().str.lower() == 'nan'))
                
                df_cleaned.loc[mask, col] = 'Uncertain'
                print(f"Replaced with 'Uncertain' in column '{col}'")
            else:
                print(f"No null/empty values found in column '{col}'")
    
    return df_cleaned

def process_results(jsonl_path, leaderboard_path, output_path):
    """
    Process results from JSONL file and create Excel output with statistics.
    
    Args:
        jsonl_path: Path to input JSONL file
        leaderboard_path: Path to existing leaderboard Excel file
        output_path: Path for output Excel file
    """
    # Read JSONL data
    data = read_jsonl_data(jsonl_path)
    results_df = pd.DataFrame(data)
    
    # Read existing leaderboard if available
    if os.path.exists(leaderboard_path):
        leaderboard_df = pd.read_excel(leaderboard_path)
        
        # Clean null values in relevant columns
        columns_to_clean = ['人工评估结果']  # Add other columns as needed
        if 'webvoyager_res' in leaderboard_df.columns:
            columns_to_clean.append('webvoyager_res')
        
        print("Checking for null/empty values in leaderboard data...")
        leaderboard_df = clean_null_values(leaderboard_df, columns_to_clean)
        
    else:
        leaderboard_df = pd.DataFrame()
    
    # Create a new column for webvoyager results if it doesn't exist
    if 'webvoyager_res' not in leaderboard_df.columns:
        leaderboard_df['webvoyager_res'] = None
    
    # Process each row in results
    for idx, result_row in results_df.iterrows():
        # Find matching rows in leaderboard
        matching_rows = leaderboard_df[
            (leaderboard_df['testcase'] == result_row['ques']) & 
            (leaderboard_df['case_name'] == result_row['web_name'])
        ]
        
        if not matching_rows.empty:
            # Get the index of the matching row
            leaderboard_idx = matching_rows.index[0]
            
            # Convert result (NO -> Fail, YES -> Pass)
            result = 'Pass' if result_row['res'].upper() == 'YES' else 'Fail'
            
            # Update the result in leaderboard
            leaderboard_df.at[leaderboard_idx, 'webvoyager_res'] = result
    
    # Clean null values again after processing (in case webvoyager_res has nulls)
    print("\nCleaning null values after processing...")
    leaderboard_df = clean_null_values(leaderboard_df, ['webvoyager_res', '人工评估结果'])
    
    # Calculate accuracy
    accuracy = calculate_accuracy(leaderboard_df)
    print(f"\nAccuracy Results:")
    
    # Define valid values for counting
    valid_values = ['Pass', 'Fail', 'Uncertain']
    valid_cases = leaderboard_df[
        leaderboard_df['webvoyager_res'].astype(str).str.strip().str.title().isin(valid_values) &
        leaderboard_df['人工评估结果'].astype(str).str.strip().str.title().isin(valid_values)
    ]
    print(f"Total valid cases: {len(valid_cases)}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Calculate and print mean human evaluation score
    if len(valid_cases) > 0:
        manual_numeric = valid_cases['人工评估结果'].astype(str).str.strip().str.title().map({
            'Pass': 1, 'Fail': 0, 'Uncertain': 0
        })
        mean_human_score = manual_numeric.mean()
        print(f"Mean human evaluation score: {mean_human_score:.4f}")
    
    # Calculate correlation between webvoyager_res and manual evaluation results
    if len(valid_cases) > 0:
        # Convert categorical data to numeric for correlation analysis
        # Pass -> 1, Fail -> 0, Uncertain -> 0
        webvoyager_numeric = valid_cases['webvoyager_res'].astype(str).str.strip().str.title().map({
            'Pass': 1, 'Fail': 0, 'Uncertain': 0
        })
        manual_numeric = valid_cases['人工评估结果'].astype(str).str.strip().str.title().map({
            'Pass': 1, 'Fail': 0, 'Uncertain': 0
        })
        
        # Check for any NaN values after conversion and remove them
        valid_mask = ~(webvoyager_numeric.isna() | manual_numeric.isna())
        webvoyager_clean = webvoyager_numeric[valid_mask]
        manual_clean = manual_numeric[valid_mask]
        
        if len(webvoyager_clean) > 1 and len(manual_clean) > 1:
            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(webvoyager_clean, manual_clean)
            
            print(f"\nCorrelation Analysis:")
            print(f"Pearson correlation coefficient: {correlation:.4f}")
            print(f"P-value: {p_value:.4f}")
            
            # Calculate additional statistics
            agreement_count = sum(webvoyager_clean == manual_clean)
            disagreement_count = len(webvoyager_clean) - agreement_count
            print(f"Agreement cases: {agreement_count}")
            print(f"Disagreement cases: {disagreement_count}")
            print(f"Agreement rate: {agreement_count/len(webvoyager_clean):.2%}")
        else:
            print(f"\nCorrelation Analysis:")
            print("Insufficient valid data for correlation analysis")
            print(f"Valid cases after cleaning: {len(webvoyager_clean)}")
    
    # Calculate statistics
    stats_df = create_statistics(results_df)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write leaderboard results
        leaderboard_df.to_excel(writer, sheet_name='Leaderboard', index=False)
        
        # Write detailed results
        results_df.to_excel(
            writer, 
            sheet_name='Detailed Results',
            index=False,
            columns=['web_name', 'id', 'ques', 'web', 'res']
        )
        
        # Write statistics
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    print(f"\nResults have been saved to: {output_path}")

def process_feature(jsonl_path, feature_leaderboard_path, output_path):
    """
    Process feature evaluation results and calculate correlation with human scores.
    
    Args:
        jsonl_path: Path to input JSONL file with agent evaluation results
        feature_leaderboard_path: Path to feature leaderboard Excel file with human scores
        output_path: Path for output Excel file
    """
    # Read JSONL data
    data = read_jsonl_data(jsonl_path)
    results_df = pd.DataFrame(data)
    
    # Calculate agent scores for each web project
    agent_scores = results_df.groupby('web_name').agg({
        'res': lambda x: sum(x.str.upper() == 'YES') / len(x)
    }).reset_index()
    agent_scores.columns = ['case_name', 'agent_score']
    
    # Read feature leaderboard with human scores
    feature_df = pd.read_excel(feature_leaderboard_path)
    
    # Clean null values in human_score_avg column
    print("Checking for null/empty values in feature leaderboard data...")
    feature_df = clean_null_values(feature_df, ['human_score_avg'])
    
    # Merge agent scores with human scores
    merged_df = pd.merge(agent_scores, feature_df[['case_name', 'human_score_avg']], 
                        on='case_name', how='inner')
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(merged_df['agent_score'], 
                                        merged_df['human_score_avg'])
    
    print(f"\nPearson Correlation Results:")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Calculate and print mean agent score
    mean_agent_score = merged_df['agent_score'].mean()
    print(f"Mean agent score: {mean_agent_score:.4f}")
    
    # Save results to Excel
    merged_df.to_excel(output_path, index=False)
    print(f"\nFeature evaluation results have been saved to: {output_path}")

def process_excel_columns(excel_path, col1_name, col2_name):
    """
    Read specific columns from Excel file and calculate accuracy and Pearson correlation.
    Only rows where both columns contain Pass/Fail/Uncertain are considered valid.
    
    Args:
        excel_path: Path to the Excel file
        col1_name: Name of the first column (e.g., 'webvoyager_res')
        col2_name: Name of the second column (e.g., '人工评估结果')
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Check if columns exist
    if col1_name not in df.columns:
        print(f"Error: Column '{col1_name}' not found in Excel file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    if col2_name not in df.columns:
        print(f"Error: Column '{col2_name}' not found in Excel file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Clean null values in the specified columns
    print("Checking for null/empty values in specified columns...")
    df = clean_null_values(df, [col1_name, col2_name])
    
    # Define valid values
    valid_values = ['Pass', 'Fail', 'Uncertain']
    
    # Filter rows where both columns contain valid values (case insensitive)
    valid_mask = (
        df[col1_name].astype(str).str.strip().str.title().isin(valid_values) &
        df[col2_name].astype(str).str.strip().str.title().isin(valid_values)
    )
    
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("Error: No valid data found. Only Pass/Fail/Uncertain values are considered valid.")
        return
    
    print(f"Found {len(valid_df)} valid rows out of {len(df)} total rows")
    
    # Calculate accuracy
    accuracy = calculate_accuracy(valid_df.rename(columns={
        col1_name: 'webvoyager_res',
        col2_name: '人工评估结果'
    }))
    
    print(f"Accuracy: {accuracy:.2%}")
    
    # Calculate Pearson correlation
    try:
        # Convert categorical data to numeric for correlation analysis
        # Pass -> 1, Fail -> 0, Uncertain -> 0 (case insensitive)
        col1_numeric = valid_df[col1_name].astype(str).str.strip().str.title().map({
            'Pass': 1, 'Fail': 0, 'Uncertain': 0
        })
        col2_numeric = valid_df[col2_name].astype(str).str.strip().str.title().map({
            'Pass': 1, 'Fail': 0, 'Uncertain': 0
        })
        
        # Check for any NaN values after conversion and remove them
        valid_mask = ~(col1_numeric.isna() | col2_numeric.isna())
        col1_clean = col1_numeric[valid_mask]
        col2_clean = col2_numeric[valid_mask]
        
        if len(col1_clean) > 1 and len(col2_clean) > 1:
            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(col1_clean, col2_clean)
            print(f"Pearson correlation: {correlation:.4f}")
            print(f"P-value: {p_value:.4f}")
            
            # Calculate and print mean human evaluation score
            mean_human_score = col2_clean.mean()
            print(f"Mean human evaluation score: {mean_human_score:.4f}")
        else:
            print("Insufficient valid data for correlation analysis")
            
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        print("Please check if the column values are in expected format (Pass/Fail/Uncertain)")

if __name__ == "__main__":
    jsonl_path = r"G:\personal\WebGen-Bench\webvoyager\realdev_res_com_qwen\refined_res.jsonl"
    leaderboard_path = r"G:\personal\WebGen-Bench\realdev\data\leaderboard.xlsx"
    output_path = r"G:\personal\WebGen-Bench\leaderboard_with_webvoyager_qwen_testcase.xlsx"
   
    process_results(jsonl_path, leaderboard_path, output_path) 
    # process_feature(jsonl_path, leaderboard_path, output_path)
    
    # Example usage of the new process_excel_columns function
    # excel_path = r"G:\personal\WebGen-Bench\leaderboard_with_browseruse_testcase.xlsx"
    
    # Uncomment the line below to run the new function
    # process_excel_columns(output_path, 'OS_Agent评估结果', '人工评估结果')