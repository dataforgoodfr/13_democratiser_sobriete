"""
extract_excel_data.py - Extract all data from Excel tables and organize them in the main tables directory

This script:
1. Reads all existing Excel files from the EWBI subdirectory
2. Extracts the data from each file 
3. Copies/moves them to the main tables directory for easier access
4. Creates a consolidated overview of all available data
"""

import pandas as pd
import os
import shutil
import sys

def extract_and_organize_excel_data():
    """Extract and organize all Excel data from subdirectories to main tables folder"""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Source directory (EWBI subfolder)
    source_ewbi_dir = os.path.join(report_dir, 'outputs', 'tables', 'EWBI')
    
    # Target directory (main tables folder)
    target_tables_dir = os.path.join(report_dir, 'outputs', 'tables')
    
    print("=== Excel Data Extraction and Organization ===")
    print(f"Source EWBI directory: {source_ewbi_dir}")
    print(f"Target tables directory: {target_tables_dir}")
    
    # Ensure target directory exists
    os.makedirs(target_tables_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(source_ewbi_dir):
        print(f"Warning: Source directory does not exist: {source_ewbi_dir}")
        print("Running the main analysis script to generate the data first...")
        
        # Import and run the main analysis
        sys.path.insert(0, current_dir)
        try:
            import ewbi_treatment
            ewbi_treatment.main()
            print("Main analysis completed. Now extracting the generated data...")
        except Exception as e:
            print(f"Error running main analysis: {e}")
            return
    
    # Get all Excel files from source directory
    if os.path.exists(source_ewbi_dir):
        excel_files = [f for f in os.listdir(source_ewbi_dir) if f.endswith('.xlsx')]
        csv_files = [f for f in os.listdir(source_ewbi_dir) if f.endswith('.csv')]
    else:
        excel_files = []
        csv_files = []
    
    all_extracted_data = []
    
    print(f"\nFound {len(excel_files)} Excel files and {len(csv_files)} CSV files in EWBI directory")
    
    # Process Excel files
    for excel_file in excel_files:
        source_path = os.path.join(source_ewbi_dir, excel_file)
        target_path = os.path.join(target_tables_dir, excel_file)
        
        print(f"\nProcessing: {excel_file}")
        
        try:
            # Copy the Excel file to target directory
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Copied to main tables directory")
            
            # Read and analyze the Excel file
            excel_data = pd.ExcelFile(source_path)
            print(f"  ✓ Sheets: {excel_data.sheet_names}")
            
            # Read the main data sheet (usually the first one with actual data)
            main_sheet = excel_data.sheet_names[0]
            df = pd.read_excel(source_path, sheet_name=main_sheet)
            
            # Store summary info
            all_extracted_data.append({
                'filename': excel_file,
                'main_sheet': main_sheet,
                'rows': len(df),
                'columns': list(df.columns),
                'indicators': df['indicator_name'].nunique() if 'indicator_name' in df.columns else 'N/A',
                'years': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A',
                'countries': df['geo'].nunique() if 'geo' in df.columns else 'N/A'
            })
            
            print(f"  ✓ Rows: {len(df)}, Columns: {len(df.columns)}")
            if 'indicator_name' in df.columns:
                print(f"  ✓ Indicators: {df['indicator_name'].nunique()}")
            if 'year' in df.columns:
                print(f"  ✓ Year range: {df['year'].min()}-{df['year'].max()}")
                
        except Exception as e:
            print(f"  ✗ Error processing {excel_file}: {e}")
    
    # Process CSV files
    for csv_file in csv_files:
        source_path = os.path.join(source_ewbi_dir, csv_file)
        target_path = os.path.join(target_tables_dir, csv_file)
        
        print(f"\nProcessing: {csv_file}")
        
        try:
            # Copy the CSV file to target directory
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Copied to main tables directory")
            
            # Read and analyze the CSV file
            df = pd.read_csv(source_path)
            print(f"  ✓ Rows: {len(df)}, Columns: {len(df.columns)}")
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {e}")
    
    # Create a summary overview file
    if all_extracted_data:
        summary_df = pd.DataFrame(all_extracted_data)
        summary_path = os.path.join(target_tables_dir, "data_summary_overview.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Created data summary overview: {summary_path}")
    
    # List all files now in the main tables directory
    print(f"\n=== Final Tables Directory Contents ===")
    if os.path.exists(target_tables_dir):
        all_files = sorted(os.listdir(target_tables_dir))
        excel_files = [f for f in all_files if f.endswith('.xlsx')]
        csv_files = [f for f in all_files if f.endswith('.csv')]
        other_files = [f for f in all_files if not f.endswith(('.xlsx', '.csv'))]
        
        print(f"Location: {target_tables_dir}")
        
        if excel_files:
            print(f"\nExcel files ({len(excel_files)}):")
            for f in excel_files:
                print(f"  • {f}")
        
        if csv_files:
            print(f"\nCSV files ({len(csv_files)}):")
            for f in csv_files:
                print(f"  • {f}")
        
        if other_files:
            print(f"\nOther files ({len(other_files)}):")
            for f in other_files:
                print(f"  • {f}")
    
    print(f"\n=== Extraction Complete ===")
    print(f"All Excel table data has been extracted and organized in:")
    print(f"{target_tables_dir}")
    print("\nThe data is now ready for analysis and external use.")

def create_consolidated_data_export():
    """Create a single consolidated Excel file with all data"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(os.path.join(current_dir, '..'))
    target_tables_dir = os.path.join(report_dir, 'outputs', 'tables')
    
    # Read all the Excel files and combine their data
    consolidated_data = {}
    
    excel_files = [
        'switzerland_vs_eu27_ewbi_level1.xlsx',
        'switzerland_vs_eu27_eu_priorities_level2.xlsx', 
        'switzerland_vs_eu27_primary_indicators.xlsx'
    ]
    
    print("\n=== Creating Consolidated Data Export ===")
    
    for excel_file in excel_files:
        file_path = os.path.join(target_tables_dir, excel_file)
        if os.path.exists(file_path):
            try:
                # Read the main data sheet
                df = pd.read_excel(file_path, sheet_name=0)
                
                # Use a shorter name for the sheet
                if 'ewbi_level1' in excel_file:
                    sheet_name = 'EWBI_Level1'
                elif 'eu_priorities_level2' in excel_file:
                    sheet_name = 'EU_Priorities_Level2'
                elif 'primary_indicators' in excel_file:
                    sheet_name = 'Primary_Indicators_Level5'
                else:
                    sheet_name = excel_file.replace('.xlsx', '')
                
                consolidated_data[sheet_name] = df
                print(f"  ✓ Added {sheet_name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  ✗ Error reading {excel_file}: {e}")
    
    # Create consolidated Excel file
    if consolidated_data:
        consolidated_path = os.path.join(target_tables_dir, "consolidated_switzerland_vs_eu27_all_data.xlsx")
        
        try:
            with pd.ExcelWriter(consolidated_path, engine='openpyxl') as writer:
                for sheet_name, df in consolidated_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add an overview sheet
                overview_data = []
                for sheet_name, df in consolidated_data.items():
                    overview_data.append({
                        'Sheet': sheet_name,
                        'Rows': len(df),
                        'Indicators': df['indicator_name'].nunique() if 'indicator_name' in df.columns else 'N/A',
                        'Years': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A',
                        'Countries': df['geo'].nunique() if 'geo' in df.columns else 'N/A'
                    })
                
                overview_df = pd.DataFrame(overview_data)
                overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            print(f"\n✓ Created consolidated file: {consolidated_path}")
            print(f"  Contains {len(consolidated_data)} data sheets plus overview")
            
        except Exception as e:
            print(f"\n✗ Error creating consolidated file: {e}")

if __name__ == "__main__":
    # Extract and organize all Excel data
    extract_and_organize_excel_data()
    
    # Create consolidated export
    create_consolidated_data_export()