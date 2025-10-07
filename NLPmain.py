import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        padding: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
    }
    .analysis-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .result-item {
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 4px solid;
        background: #f8f9fa;
        color: #2c3e50;
        font-size: 0.95rem;
    }
    .result-high {
        border-left-color: #e74c3c;
        background: #fdEDEC;
    }
    .result-medium {
        border-left-color: #f39c12;
        background: #fef5e7;
    }
    .result-low {
        border-left-color: #27ae60;
        background: #eafaf1;
    }
    .result-info {
        border-left-color: #3498db;
        background: #ebf5fb;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def detect_semantic_issues(df):
    """Detect semantic issues in the data"""
    issues = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Detect potential semantic columns
        semantic_keywords = ['name', 'title', 'description', 'comment', 'note', 'label', 'category', 'type']
        if any(keyword in col_lower for keyword in semantic_keywords):
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:
                    issues.append({
                        'type': 'high',
                        'message': f"High cardinality in semantic column '{col}': {df[col].nunique()} unique values found"
                    })
        
        # Check for mixed data types
        if df[col].dtype == 'object':
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count > 0 and numeric_count < len(df):
                issues.append({
                    'type': 'medium',
                    'message': f"Mixed data types in '{col}': Contains both text and numeric values"
                })
    
    return issues

def detect_manipulation_patterns(df):
    """Detect potential data manipulation patterns"""
    patterns = []
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        # Check for rounded numbers
        rounded_count = sum(df[col].apply(lambda x: x == round(x) if pd.notnull(x) else False))
        rounded_ratio = rounded_count / len(df)
        
        if rounded_ratio > 0.9:
            patterns.append({
                'type': 'medium',
                'message': f"Rounded values in '{col}': {rounded_ratio:.1%} of values are rounded"
            })
        
        # Check skewness
        if len(df[col].dropna()) > 0:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                patterns.append({
                    'type': 'high',
                    'message': f"High skewness in '{col}': Value = {skewness:.2f} (potential outliers)"
                })
    
    return patterns

def detect_clickbait_patterns(df):
    """Detect potential clickbait patterns in text columns"""
    clickbait_indicators = []
    text_columns = df.select_dtypes(include=['object']).columns
    
    clickbait_keywords = [
        'shocking', 'amazing', 'unbelievable', 'secret', 'revealed', 
        'you wont believe', 'what happened next', 'goes viral',
        'everyone is talking about', 'breaking', 'urgent', 'must see',
        'will blow your mind', 'this is why', 'the truth about'
    ]
    
    for col in text_columns:
        sample_texts = df[col].dropna().head(100)
        clickbait_count = 0
        
        for text in sample_texts:
            if isinstance(text, str):
                text_lower = text.lower()
                for keyword in clickbait_keywords:
                    if keyword in text_lower:
                        clickbait_count += 1
                        break
        
        if clickbait_count > 5:
            clickbait_indicators.append({
                'type': 'high',
                'message': f"Clickbait language in '{col}': {clickbait_count} instances detected"
            })
        elif clickbait_count > 0:
            clickbait_indicators.append({
                'type': 'medium',
                'message': f"Some clickbait language in '{col}': {clickbait_count} instances found"
            })
    
    return clickbait_indicators

def detect_satire_indicators(df):
    """Detect potential satire indicators"""
    satire_signals = []
    text_columns = df.select_dtypes(include=['object']).columns
    
    satire_keywords = [
        'satire', 'parody', 'humor', 'comedy', 'joke', 'not real',
        'fictional', 'fake news', 'entertainment purposes', 'just kidding'
    ]
    
    extreme_indicators = [
        'absolutely', 'completely', 'totally', 'utterly', '100%', 
        'worst ever', 'best ever', 'never before', 'unprecedented'
    ]
    
    for col in text_columns:
        sample_texts = df[col].dropna().head(50)
        satire_count = 0
        extreme_count = 0
        
        for text in sample_texts:
            if isinstance(text, str):
                text_lower = text.lower()
                
                for keyword in satire_keywords:
                    if keyword in text_lower:
                        satire_count += 1
                        break
                
                for word in extreme_indicators:
                    if word in text_lower:
                        extreme_count += 1
        
        if satire_count > 0:
            satire_signals.append({
                'type': 'high',
                'message': f"Satire indicators in '{col}': {satire_count} explicit markers found"
            })
        
        if extreme_count > 10:
            satire_signals.append({
                'type': 'medium',
                'message': f"Extreme language in '{col}': {extreme_count} instances detected"
            })
    
    return satire_signals

def main():
    # Header
    st.markdown('<h1 class="main-header">Advanced Data Analysis Platform</h1>', unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload CSV File for Analysis", 
        type=['csv'],
        help="Upload your CSV file to perform comprehensive data analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display success message
            st.success(f"File loaded successfully: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            
            # Single page layout with two main sections
            display_data_overview(df)
            display_content_analysis(df)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a CSV file to begin analysis")
        
        # Sample data structure
        with st.expander("View Expected Data Format"):
            sample_data = {
                'Customer_Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Davis'],
                'Age': [25, 30, 35, 28],
                'Annual_Salary': [50000, 60000, 70000, 55000],
                'Department': ['IT', 'HR', 'IT', 'Finance'],
                'Performance_Score': [85, 92, 78, 88]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)

def display_data_overview(df):
    st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-box"><h4>Total Records</h4><h3>{df.shape[0]:,}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h4>Total Columns</h4><h3>{df.shape[1]}</h3></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h4>Missing Values</h4><h3>{df.isnull().sum().sum()}</h3></div>', unsafe_allow_html=True)
    with col4:
        mem_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f'<div class="metric-box"><h4>Memory Usage</h4><h3>{mem_usage:.1f} MB</h3></div>', unsafe_allow_html=True)
    
    # Data preview
    st.subheader("Data Preview")
    
    preview_tabs = st.tabs(["First 10 Rows", "Last 10 Rows", "Column Information"])
    
    with preview_tabs[0]:
        st.dataframe(df.head(10), use_container_width=True)
    
    with preview_tabs[1]:
        st.dataframe(df.tail(10), use_container_width=True)
    
    with preview_tabs[2]:
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

def display_content_analysis(df):
    st.markdown('<h2 class="section-header">Content Analysis</h2>', unsafe_allow_html=True)
    
    # Analysis type dropdown
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        [
            "All Analyses",
            "Semantic Analysis", 
            "Data Manipulation Detection",
            "Clickbait Language Detection", 
            "Satire and Extreme Language Analysis"
        ],
        help="Choose which analysis to run on your data"
    )
    
    # Run selected analysis
    if analysis_type == "All Analyses":
        st.info("Running all content analyses...")
        run_all_analyses(df)
    elif analysis_type == "Semantic Analysis":
        run_semantic_analysis(df)
    elif analysis_type == "Data Manipulation Detection":
        run_manipulation_analysis(df)
    elif analysis_type == "Clickbait Language Detection":
        run_clickbait_analysis(df)
    elif analysis_type == "Satire and Extreme Language Analysis":
        run_satire_analysis(df)

def run_all_analyses(df):
    """Run and display all analyses"""
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("Running semantic analysis..."):
            semantic_results = detect_semantic_issues(df)
        
        with st.spinner("Detecting data manipulation..."):
            manipulation_results = detect_manipulation_patterns(df)
    
    with col2:
        with st.spinner("Scanning for clickbait patterns..."):
            clickbait_results = detect_clickbait_patterns(df)
        
        with st.spinner("Checking for satire indicators..."):
            satire_results = detect_satire_indicators(df)
    
    # Display all results
    display_analysis_results("Semantic Analysis", semantic_results, 
                           "Identifies semantic issues in data structure and content")
    
    display_analysis_results("Data Manipulation Detection", manipulation_results,
                           "Detects patterns that may indicate data manipulation")
    
    display_analysis_results("Clickbait Language Detection", clickbait_results,
                           "Identifies sensational language patterns")
    
    display_analysis_results("Satire and Extreme Language Analysis", satire_results,
                           "Detects satire indicators and extreme language patterns")
    
    # Summary
    total_findings = len(semantic_results) + len(manipulation_results) + len(clickbait_results) + len(satire_results)
    display_analysis_summary(total_findings)

def run_semantic_analysis(df):
    """Run and display semantic analysis only"""
    with st.spinner("Running semantic analysis..."):
        results = detect_semantic_issues(df)
    
    display_analysis_results("Semantic Analysis", results,
                           "Identifies semantic issues in data structure and content")
    display_analysis_summary(len(results))

def run_manipulation_analysis(df):
    """Run and display manipulation analysis only"""
    with st.spinner("Detecting data manipulation patterns..."):
        results = detect_manipulation_patterns(df)
    
    display_analysis_results("Data Manipulation Detection", results,
                           "Detects patterns that may indicate data manipulation")
    display_analysis_summary(len(results))

def run_clickbait_analysis(df):
    """Run and display clickbait analysis only"""
    with st.spinner("Scanning for clickbait patterns..."):
        results = detect_clickbait_patterns(df)
    
    display_analysis_results("Clickbait Language Detection", results,
                           "Identifies sensational language patterns")
    display_analysis_summary(len(results))

def run_satire_analysis(df):
    """Run and display satire analysis only"""
    with st.spinner("Checking for satire indicators..."):
        results = detect_satire_indicators(df)
    
    display_analysis_results("Satire and Extreme Language Analysis", results,
                           "Detects satire indicators and extreme language patterns")
    display_analysis_summary(len(results))

def display_analysis_results(title, results, description):
    """Display analysis results in a consistent format"""
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.subheader(title)
    st.write(description)
    
    if results:
        for result in results:
            css_class = f"result-{result['type']}"
            st.markdown(f'<div class="result-item {css_class}">{result["message"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-item result-info">No issues detected in this analysis</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_analysis_summary(total_findings):
    """Display analysis summary"""
    st.subheader("Analysis Summary")
    
    if total_findings == 0:
        st.success("No issues detected in the analysis.")
    else:
        st.info(f"Found {total_findings} potential issue(s) in the analysis")

if __name__ == "__main__":
    main()
