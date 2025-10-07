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
    page_title="Advanced CSV File Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #28a745;
    }
    .insight-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #ffc107;
    }
    .insight-danger {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def detect_semantic_issues(df):
    """Detect semantic issues in the data"""
    issues = []
    
    # Check for columns that might be identifiers but have semantic meaning
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Detect potential semantic columns
        semantic_keywords = ['name', 'title', 'description', 'comment', 'note', 'label', 'category', 'type']
        if any(keyword in col_lower for keyword in semantic_keywords):
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:
                    issues.append(f"High cardinality in semantic column '{col}': {df[col].nunique()} unique values")
    
    return issues

def detect_manipulation_patterns(df):
    """Detect potential data manipulation patterns"""
    patterns = []
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        # Check for rounded numbers (potential manipulation)
        rounded_count = sum(df[col].apply(lambda x: x == round(x) if pd.notnull(x) else False))
        rounded_ratio = rounded_count / len(df)
        
        if rounded_ratio > 0.9:
            patterns.append(f"Column '{col}' has {rounded_ratio:.1%} rounded values - possible data smoothing")
        
        # Check for Benford's Law violations in first digits
        if len(df) > 100:
            first_digits = df[col].dropna().apply(lambda x: int(str(abs(x))[0]) if x != 0 else 0)
            digit_counts = first_digits.value_counts().sort_index()
            if len(digit_counts) > 0:
                benford_expected = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
                actual_proportions = digit_counts / len(first_digits)
                
                # Simple Benford's law check
                if actual_proportions.get(1, 0) < 0.25:
                    patterns.append(f"Column '{col}' shows potential Benford's Law deviation")
    
    return patterns

def detect_clickbait_patterns(df):
    """Detect potential clickbait patterns in text columns"""
    clickbait_indicators = []
    text_columns = df.select_dtypes(include=['object']).columns
    
    clickbait_keywords = [
        'shocking', 'amazing', 'unbelievable', 'secret', 'revealed', 
        'you wont believe', 'what happened next', 'goes viral',
        'everyone is talking about', 'breaking', 'urgent'
    ]
    
    for col in text_columns:
        # Sample some text data for analysis
        sample_texts = df[col].dropna().head(100)
        
        for text in sample_texts:
            if isinstance(text, str):
                text_lower = text.lower()
                for keyword in clickbait_keywords:
                    if keyword in text_lower:
                        clickbait_indicators.append(f"Potential clickbait language in '{col}': contains '{keyword}'")
                        break
    
    return list(set(clickbait_indicators))[:5]  # Return top 5 unique indicators

def detect_satire_indicators(df):
    """Detect potential satire indicators"""
    satire_signals = []
    text_columns = df.select_dtypes(include=['object']).columns
    
    satire_keywords = [
        'satire', 'parody', 'humor', 'comedy', 'joke', 'not real',
        'fictional', 'fake news', 'entertainment purposes'
    ]
    
    extreme_value_indicators = [
        'absolutely', 'completely', 'totally', 'utterly', '100%', 
        'worst ever', 'best ever', 'never before', 'unprecedented'
    ]
    
    for col in text_columns:
        sample_texts = df[col].dropna().head(50)
        
        for text in sample_texts:
            if isinstance(text, str):
                text_lower = text.lower()
                
                # Check for explicit satire disclaimers
                for keyword in satire_keywords:
                    if keyword in text_lower:
                        satire_signals.append(f"Explicit satire indicator in '{col}': '{keyword}'")
                        break
                
                # Check for extreme language
                extreme_count = sum(1 for word in extreme_value_indicators if word in text_lower)
                if extreme_count >= 3:
                    satire_signals.append(f"Extreme language pattern in '{col}' - potential satire")
    
    return list(set(satire_signals))[:5]

def main():
    # Header
    st.markdown('<h1 class="main-header">Advanced CSV Data Analysis Platform</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<h2 class="section-header">Data File Upload</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select CSV file for comprehensive analysis", 
        type=['csv'],
        help="Upload your CSV file for detailed data analysis and quality assessment"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display success message
            st.success(f"File successfully loaded! Dataset contains {df.shape[0]:,} records and {df.shape[1]:,} attributes")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Data Overview", 
                "Data Quality", 
                "Statistical Analysis", 
                "Data Visualizations",
                "Content Analysis",
                "Advanced Insights",
                "Export Results"
            ])
            
            with tab1:
                display_data_overview(df)
                
            with tab2:
                display_data_quality(df)
                
            with tab3:
                display_statistical_analysis(df)
                
            with tab4:
                display_visualizations(df)
                
            with tab5:
                display_content_analysis(df)
                
            with tab6:
                display_advanced_insights(df)
                
            with tab7:
                display_export_options(df)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a CSV file to begin the analysis process")
        
        # Sample data structure
        st.markdown("### Expected Data Structure")
        sample_data = {
            'Customer_Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Davis'],
            'Age': [25, 30, 35, 28],
            'Annual_Salary': [50000, 60000, 70000, 55000],
            'Department': ['Information Technology', 'Human Resources', 'IT Support', 'Finance'],
            'Performance_Score': [85, 92, 78, 88]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

def display_data_overview(df):
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Attributes", df.shape[1])
    with col3:
        st.metric("Total Data Points", f"{df.shape[0] * df.shape[1]:,}")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data preview
    st.subheader("Data Preview")
    
    preview_option = st.radio(
        "Select preview type:",
        ["First 10 records", "Last 10 records", "Random sample"],
        horizontal=True
    )
    
    if preview_option == "First 10 records":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_option == "Last 10 records":
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.dataframe(df.sample(10), use_container_width=True)
    
    # Column information
    st.subheader("Attribute Details")
    col_info = pd.DataFrame({
        'Attribute Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Values': df.count().values,
        'Missing Values': df.isnull().sum().values
    })
    st.dataframe(col_info, use_container_width=True)

def display_data_quality(df):
    st.markdown('<h2 class="section-header">Data Quality Assessment</h2>', unsafe_allow_html=True)
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    
    missing_data = pd.DataFrame({
        'Attribute': df.columns,
        'Missing Values': df.isnull().sum().values,
        'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(missing_data, use_container_width=True)
    
    with col2:
        # Missing values summary
        total_missing = df.isnull().sum().sum()
        columns_with_missing = (df.isnull().sum() > 0).sum()
        
        st.metric("Total Missing Values", total_missing)
        st.metric("Attributes with Missing Data", columns_with_missing)
        st.metric("Complete Records", len(df) - df.isnull().any(axis=1).sum())
    
    # Duplicates analysis
    st.subheader("Duplicate Records Analysis")
    duplicate_rows = df.duplicated().sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duplicate Records", duplicate_rows)
    with col2:
        st.metric("Duplicate Percentage", f"{(duplicate_rows / len(df) * 100):.2f}%")
    
    if duplicate_rows > 0:
        if st.button("Display Duplicate Records"):
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)

def display_statistical_analysis(df):
    st.markdown('<h2 class="section-header">Statistical Analysis</h2>', unsafe_allow_html=True)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numerical_cols:
        st.subheader("Numerical Attributes Summary")
        
        # Let user select which numerical column to analyze
        selected_num_col = st.selectbox("Select numerical attribute for detailed analysis:", numerical_cols)
        
        if selected_num_col:
            col_stats = df[selected_num_col].describe()
            
            # Display statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{col_stats['mean']:.2f}")
                st.metric("Standard Deviation", f"{col_stats['std']:.2f}")
            
            with col2:
                st.metric("Minimum", f"{col_stats['min']:.2f}")
                st.metric("25th Percentile", f"{col_stats['25%']:.2f}")
            
            with col3:
                st.metric("Median", f"{col_stats['50%']:.2f}")
                st.metric("75th Percentile", f"{col_stats['75%']:.2f}")
            
            with col4:
                st.metric("Maximum", f"{col_stats['max']:.2f}")
                st.metric("Valid Count", f"{col_stats['count']:.0f}")
        
        # Correlation matrix for numerical columns
        if len(numerical_cols) > 1:
            st.subheader("Attribute Correlation Matrix")
            corr_matrix = df[numerical_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix of Numerical Attributes')
            st.pyplot(fig)
    
    if categorical_cols:
        st.subheader("Categorical Attributes Summary")
        selected_cat_col = st.selectbox("Select categorical attribute for analysis:", categorical_cols)
        
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Value Distribution:")
                st.dataframe(value_counts, use_container_width=True)
            
            with col2:
                st.write("Distribution Statistics:")
                st.metric("Unique Categories", value_counts.nunique())
                st.metric("Most Frequent Category", value_counts.index[0])
                st.metric("Mode Frequency", value_counts.iloc[0])

def display_visualizations(df):
    st.markdown('<h2 class="section-header">Data Visualization</h2>', unsafe_allow_html=True)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Distribution Histogram", "Box Plot Analysis", "Scatter Plot", "Category Distribution", "Pie Chart"]
    )
    
    if viz_type == "Distribution Histogram" and numerical_cols:
        col = st.selectbox("Select numerical attribute:", numerical_cols)
        if col:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[col].hist(bins=30, ax=ax, edgecolor='black')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    
    elif viz_type == "Box Plot Analysis" and numerical_cols:
        col = st.selectbox("Select numerical attribute:", numerical_cols)
        if col:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=col, ax=ax)
            ax.set_title(f'Box Plot Analysis of {col}')
            st.pyplot(fig)
    
    elif viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis attribute:", numerical_cols)
        with col2:
            y_col = st.selectbox("Y-axis attribute:", numerical_cols)
        
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[x_col], df[y_col], alpha=0.6, color='steelblue')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Relationship Analysis: {x_col} vs {y_col}')
            st.pyplot(fig)
    
    elif viz_type == "Category Distribution" and categorical_cols:
        col = st.selectbox("Select categorical attribute:", categorical_cols)
        if col:
            top_n = st.slider("Number of top categories to display:", 5, 20, 10)
            value_counts = df[col].value_counts().head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title(f'Top {top_n} Categories in {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    elif viz_type == "Pie Chart" and categorical_cols:
        col = st.selectbox("Select categorical attribute:", categorical_cols)
        if col:
            top_n = st.slider("Number of top categories to display:", 3, 10, 5)
            value_counts = df[col].value_counts().head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Category Distribution for {col} (Top {top_n})')
            st.pyplot(fig)

def display_content_analysis(df):
    st.markdown('<h2 class="section-header">Content Pattern Analysis</h2>', unsafe_allow_html=True)
    
    # Run all detection functions
    semantic_issues = detect_semantic_issues(df)
    manipulation_patterns = detect_manipulation_patterns(df)
    clickbait_indicators = detect_clickbait_patterns(df)
    satire_indicators = detect_satire_indicators(df)
    
    # Display results in expandable sections
    with st.expander("Semantic Analysis Results", expanded=True):
        if semantic_issues:
            for issue in semantic_issues:
                st.markdown(f'<div class="insight-warning">{issue}</div>', unsafe_allow_html=True)
        else:
            st.info("No significant semantic issues detected")
    
    with st.expander("Data Manipulation Indicators"):
        if manipulation_patterns:
            for pattern in manipulation_patterns:
                st.markdown(f'<div class="insight-danger">{pattern}</div>', unsafe_allow_html=True)
        else:
            st.info("No obvious data manipulation patterns detected")
    
    with st.expander("Clickbait Language Detection"):
        if clickbait_indicators:
            for indicator in clickbait_indicators:
                st.markdown(f'<div class="insight-warning">{indicator}</div>', unsafe_allow_html=True)
        else:
            st.info("No clickbait language patterns detected")
    
    with st.expander("Satire and Extreme Language Analysis"):
        if satire_indicators:
            for indicator in satire_indicators:
                st.markdown(f'<div class="insight-warning">{indicator}</div>', unsafe_allow_html=True)
        else:
            st.info("No satire indicators or extreme language patterns detected")

def display_advanced_insights(df):
    st.markdown('<h2 class="section-header">Advanced Data Insights</h2>', unsafe_allow_html=True)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    insights = []
    
    # Data quality insights
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        insights.append(('warning', f"Data Quality Alert: {len(missing_cols)} attributes contain missing values requiring attention"))
    
    if df.duplicated().sum() > 0:
        insights.append(('danger', f"Data Integrity Issue: {df.duplicated().sum()} duplicate records identified"))
    
    # Numerical insights
    for col in numerical_cols:
        if df[col].nunique() == 1:
            insights.append(('warning', f"Constant Value Attribute: '{col}' contains only a single unique value"))
        
        # Check for potential outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:  # Avoid division by zero
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].shape[0]
            if outliers > 0:
                insights.append(('warning', f"Statistical Outliers: '{col}' contains {outliers} potential outlier values"))
    
    # Categorical insights
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count == len(df):
            insights.append(('positive', f"Unique Identifier: '{col}' appears to function as a unique record identifier"))
        elif unique_count == 1:
            insights.append(('warning', f"Uniform Category: '{col}' contains only a single category value"))
        elif unique_count < 10:
            insights.append(('positive', f"Low Cardinality: '{col}' has manageable {unique_count} unique categories"))
    
    # Display insights
    if insights:
        st.subheader("Key Analytical Findings")
        for insight_type, insight in insights:
            if insight_type == 'positive':
                st.markdown(f'<div class="insight-positive">{insight}</div>', unsafe_allow_html=True)
            elif insight_type == 'warning':
                st.markdown(f'<div class="insight-warning">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-danger">{insicontinueght}</div>', unsafe_allow_html=True)
    else:
        st.info("No significant analytical insights detected. The dataset appears well-structured.")
    
    # Data summary
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Dataset Structure**")
        st.write(f"Total Records: {df.shape[0]:,}")
        st.write(f"Total Attributes: {df.shape[1]}")
    
    with col2:
        st.write("**Data Composition**")
        st.write(f"Numerical Attributes: {len(numerical_cols)}")
        st.write(f"Categorical Attributes: {len(categorical_cols)}")
    
    with col3:
        st.write("**Quality Metrics**")
        st.write(f"Missing Values: {df.isnull().sum().sum()}")
        st.write(f"Duplicate Records: {df.duplicated().sum()}")

def display_export_options(df):
    st.markdown('<h2 class="section-header">Analysis Results Export</h2>', unsafe_allow_html=True)
    
    # Export data summary
    st.subheader("Export Analysis Summary")
    
    # Generate comprehensive summary statistics
    summary_data = {
        'Analysis Metric': [
            'Total Records', 'Total Attributes', 'Total Missing Values',
            'Total Duplicate Records', 'Memory Usage (MB)',
            'Numerical Attributes', 'Categorical Attributes'
        ],
        'Value': [
            df.shape[0], df.shape[1], df.isnull().sum().sum(),
            df.duplicated().sum(), f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
            len(df.select_dtypes(include=[np.number]).columns),
            len(df.select_dtypes(include=['object']).columns)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Convert to CSV
    csv_summary = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis Summary",
        data=csv_summary,
        file_name="comprehensive_analysis_summary.csv",
        mime="text/csv"
    )
    
    # Export cleaned data
    st.subheader("Export Processed Dataset")
    
    # Data processing options
    st.write("Data Processing Options:")
    remove_duplicates = st.checkbox("Remove duplicate records")
    fill_missing = st.checkbox("Handle missing values")
    normalize_text = st.checkbox("Normalize text data (trim whitespace)")
    
    processed_df = df.copy()
    
    if remove_duplicates:
        processed_df = processed_df.drop_duplicates()
        st.write(f"Removed {df.shape[0] - processed_df.shape[0]} duplicate records")
    
    if fill_missing:
        # Fill numerical columns with median, categorical with mode
        for col in processed_df.columns:
            if processed_df[col].dtype in [np.number]:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            else:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Not Specified")
        st.write("Missing values have been imputed")
    
    if normalize_text:
        text_cols = processed_df.select_dtypes(include=['object']).columns
        for col in text_cols:
            processed_df[col] = processed_df[col].astype(str).str.strip()
        st.write("Text data normalized")
    
    # Display processed data info
    st.write(f"Processed Dataset Structure: {processed_df.shape[0]} records × {processed_df.shape[1]} attributes")
    
    # Download processed data
    csv_processed = processed_df.to_csv(index=False)
    st.download_button(
        label="Download Processed Data",
        data=csv_processed,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
