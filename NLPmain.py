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
    page_title="CSV File Analyzer",
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
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 CSV File Analyzer</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<h2 class="section-header">📁 Upload Your CSV File</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your CSV file for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display success message
            st.success(f"✅ File successfully loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📋 Data Overview", 
                "🔍 Data Quality", 
                "📈 Statistical Analysis", 
                "📊 Visualizations",
                "🔮 Insights",
                "💾 Export"
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
                display_insights(df)
                
            with tab6:
                display_export_options(df)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        # Display instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        # Sample data structure
        st.markdown("### 📝 Expected File Format")
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Age': [25, 30, 35, 28],
            'Salary': [50000, 60000, 70000, 55000],
            'Department': ['IT', 'HR', 'IT', 'Finance']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

def display_data_overview(df):
    st.markdown('<h2 class="section-header">📋 Data Overview</h2>', unsafe_allow_html=True)
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Total Cells", df.shape[0] * df.shape[1])
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data preview
    st.subheader("Data Preview")
    
    preview_option = st.radio(
        "Preview Type:",
        ["First 10 rows", "Last 10 rows", "Random 10 rows"],
        horizontal=True
    )
    
    if preview_option == "First 10 rows":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_option == "Last 10 rows":
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.dataframe(df.sample(10), use_container_width=True)
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(col_info, use_container_width=True)

def display_data_quality(df):
    st.markdown('<h2 class="section-header">🔍 Data Quality Analysis</h2>', unsafe_allow_html=True)
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
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
        st.metric("Columns with Missing Values", columns_with_missing)
        st.metric("Complete Cases", len(df) - df.isnull().any(axis=1).sum())
    
    # Duplicates analysis
    st.subheader("Duplicate Analysis")
    duplicate_rows = df.duplicated().sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duplicate Rows", duplicate_rows)
    with col2:
        st.metric("Duplicate Percentage", f"{(duplicate_rows / len(df) * 100):.2f}%")
    
    if duplicate_rows > 0:
        if st.button("Show Duplicate Rows"):
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)

def display_statistical_analysis(df):
    st.markdown('<h2 class="section-header">📈 Statistical Analysis</h2>', unsafe_allow_html=True)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numerical_cols:
        st.subheader("Numerical Columns Summary")
        
        # Let user select which numerical column to analyze
        selected_num_col = st.selectbox("Select numerical column for detailed analysis:", numerical_cols)
        
        if selected_num_col:
            col_stats = df[selected_num_col].describe()
            
            # Display statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{col_stats['mean']:.2f}")
                st.metric("Std Dev", f"{col_stats['std']:.2f}")
            
            with col2:
                st.metric("Min", f"{col_stats['min']:.2f}")
                st.metric("25%", f"{col_stats['25%']:.2f}")
            
            with col3:
                st.metric("50%", f"{col_stats['50%']:.2f}")
                st.metric("75%", f"{col_stats['75%']:.2f}")
            
            with col4:
                st.metric("Max", f"{col_stats['max']:.2f}")
                st.metric("Count", f"{col_stats['count']:.0f}")
        
        # Correlation matrix for numerical columns
        if len(numerical_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numerical_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
    
    if categorical_cols:
        st.subheader("Categorical Columns Summary")
        selected_cat_col = st.selectbox("Select categorical column for analysis:", categorical_cols)
        
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Value Counts:")
                st.dataframe(value_counts, use_container_width=True)
            
            with col2:
                st.write("Basic Stats:")
                st.metric("Unique Values", value_counts.nunique())
                st.metric("Most Frequent", value_counts.index[0])
                st.metric("Mode Frequency", value_counts.iloc[0])

def display_visualizations(df):
    st.markdown('<h2 class="section-header">📊 Data Visualizations</h2>', unsafe_allow_html=True)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Pie Chart"]
    )
    
    if viz_type == "Histogram" and numerical_cols:
        col = st.selectbox("Select numerical column:", numerical_cols)
        if col:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[col].hist(bins=30, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    
    elif viz_type == "Box Plot" and numerical_cols:
        col = st.selectbox("Select numerical column:", numerical_cols)
        if col:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=col, ax=ax)
            ax.set_title(f'Box Plot of {col}')
            st.pyplot(fig)
    
    elif viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", numerical_cols)
        with col2:
            y_col = st.selectbox("Y-axis:", numerical_cols)
        
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            st.pyplot(fig)
    
    elif viz_type == "Bar Chart" and categorical_cols:
        col = st.selectbox("Select categorical column:", categorical_cols)
        if col:
            top_n = st.slider("Number of top categories to show:", 5, 20, 10)
            value_counts = df[col].value_counts().head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Top {top_n} Categories in {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    elif viz_type == "Pie Chart" and categorical_cols:
        col = st.selectbox("Select categorical column:", categorical_cols)
        if col:
            top_n = st.slider("Number of top categories to show:", 3, 10, 5)
            value_counts = df[col].value_counts().head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Distribution of {col} (Top {top_n})')
            st.pyplot(fig)

def display_insights(df):
    st.markdown('<h2 class="section-header">🔮 Automated Insights</h2>', unsafe_allow_html=True)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    insights = []
    
    # Data quality insights
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        insights.append(f"🚨 **Data Quality Issue**: {len(missing_cols)} columns have missing values")
    
    if df.duplicated().sum() > 0:
        insights.append(f"⚠️ **Duplicate Data**: {df.duplicated().sum()} duplicate rows found")
    
    # Numerical insights
    for col in numerical_cols:
        if df[col].nunique() == 1:
            insights.append(f"ℹ️ **Constant Column**: '{col}' has only one unique value")
        
        # Check for potential outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].shape[0]
        
        if outliers > 0:
            insights.append(f"📊 **Outliers Detected**: '{col}' has {outliers} potential outliers")
    
    # Categorical insights
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count == len(df):
            insights.append(f"🔑 **Unique Identifier**: '{col}' appears to be a unique identifier")
        elif unique_count == 1:
            insights.append(f"📝 **Constant Category**: '{col}' has only one category")
        elif unique_count < 10:
            insights.append(f"🏷️ **Low Cardinality**: '{col}' has only {unique_count} unique categories")
    
    # Display insights
    if insights:
        st.subheader("Key Findings")
        for insight in insights:
            st.write(insight)
    else:
        st.info("No significant insights detected. Your data looks clean!")
    
    # Data summary
    st.subheader("Quick Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Data Shape**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
    
    with col2:
        st.write("**Data Types**")
        st.write(f"Numerical: {len(numerical_cols)}")
        st.write(f"Categorical: {len(categorical_cols)}")
    
    with col3:
        st.write("**Data Quality**")
        st.write(f"Missing Values: {df.isnull().sum().sum()}")
        st.write(f"Duplicate Rows: {df.duplicated().sum()}")

def display_export_options(df):
    st.markdown('<h2 class="section-header">💾 Export Analysis</h2>', unsafe_allow_html=True)
    
    # Export data summary
    st.subheader("Data Summary Export")
    
    # Generate summary statistics
    summary_data = {
        'Metric': [
            'Total Rows', 'Total Columns', 'Total Missing Values',
            'Total Duplicate Rows', 'Memory Usage (MB)'
        ],
        'Value': [
            df.shape[0], df.shape[1], df.isnull().sum().sum(),
            df.duplicated().sum(), f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Convert to CSV
    csv_summary = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Summary Statistics",
        data=csv_summary,
        file_name="data_summary.csv",
        mime="text/csv"
    )
    
    # Export cleaned data
    st.subheader("Export Cleaned Data")
    
    # Data cleaning options
    st.write("Data Cleaning Options:")
    remove_duplicates = st.checkbox("Remove duplicate rows")
    fill_missing = st.checkbox("Fill missing values")
    
    cleaned_df = df.copy()
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        # Fill numerical columns with mean, categorical with mode
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in [np.number]:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown")
    
    # Display cleaned data info
    st.write(f"Cleaned Data Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    
    # Download cleaned data
    csv_cleaned = cleaned_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv_cleaned,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
