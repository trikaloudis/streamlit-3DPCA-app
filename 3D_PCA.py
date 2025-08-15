# main.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="3D PCA Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
def load_data(file):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        # Assume the first column is the index
        return pd.read_csv(file, index_col=0)
    except Exception as e:
        st.error(f"Error loading data file: {e}")
        return None

# --- Main App UI ---
st.title("🔬 AquOmixLab - 3D Principal Component Analysis (PCA) Explorer")

st.markdown("""
Welcome to the 3D PCA Explorer! This app allows you to perform a Principal Component Analysis on your dataset and visualize the results in an interactive 3D plot.

**How to use this app:**
1.  **Upload your data file:** This CSV should have features as rows and samples as columns. The first column must contain the feature IDs.
2.  **Upload your metadata file:** This CSV should have samples as rows and their attributes (e.g., group, condition) as columns. The first column must contain the sample IDs.
3.  **Select an attribute:** Choose a metadata column from the sidebar to color-code the samples in the plot.
4.  **Explore the plots:** Interact with the 3D plots by rotating, zooming, and hovering over points to see more details.
""")

# --- Sidebar for File Uploads and Controls ---
with st.sidebar:
    st.header("⚙️ Controls & Settings")

    # File uploader for the main data
    data_file = st.file_uploader(
        "1. Upload Your Data CSV",
        type=["csv"],
        help="CSV file where rows are features and columns are samples. The first column should be the feature ID."
    )

    # File uploader for the metadata
    metadata_file = st.file_uploader(
        "2. Upload Your Metadata CSV",
        type=["csv"],
        help="CSV file where rows are samples and columns are attributes. The first column should be the sample ID."
    )
# Placeholder for the attribute selection dropdown
    attribute_selector_placeholder = st.empty()

    # Add a divider for visual separation before the logo
    st.divider()

    # Display the logo and hyperlink at the bottom of the sidebar
    st.image("Aquomixlab Logo v2 white font.jpg")
    st.markdown("[www.aquomixlab.com](https://www.aquomixlab.com/)")
    
    # Placeholder for the attribute selection dropdown
    attribute_selector_placeholder = st.empty()


# --- Main Panel for Analysis and Visualization ---
if data_file is not None and metadata_file is not None:
    # Load data and metadata
    data_df = load_data(data_file)
    metadata_df = load_data(metadata_file)

    if data_df is not None and metadata_df is not None:
        st.success("✅ Data and metadata files loaded successfully!")

        # --- Data Preprocessing and Validation ---
        try:
            # Transpose the data so that samples are rows and features are columns
            data_df_transposed = data_df.T

            # Verify that sample names match between data and metadata
            if not all(data_df_transposed.index == metadata_df.index):
                st.warning(
                    "⚠️ Sample names in data and metadata files do not match or are not in the same order. "
                    "Attempting to align them based on sample IDs..."
                )
                # Align metadata with data
                metadata_df = metadata_df.reindex(data_df_transposed.index)
                if metadata_df.isnull().values.any():
                     st.error("❌ Alignment failed. Please ensure sample IDs are consistent between files.")
                     st.stop()
                else:
                    st.info("✅ Sample alignment successful.")


            # --- PCA Calculation ---
            # Separate features from metadata for scaling
            features = data_df_transposed.values

            # Standardize the features (important for PCA)
            features_scaled = StandardScaler().fit_transform(features)

            # Perform PCA
            pca = PCA(n_components=3)
            principal_components = pca.fit_transform(features_scaled)

            # Create a DataFrame with the PCA results
            pca_df = pd.DataFrame(
                data=principal_components,
                columns=['PC1', 'PC2', 'PC3'],
                index=data_df_transposed.index
            )

            # Combine PCA results with metadata for plotting
            final_df = pd.concat([pca_df, metadata_df], axis=1)

            # --- Sidebar Attribute Selector ---
            # Populate the attribute selector now that metadata is loaded
            grouping_attribute = attribute_selector_placeholder.selectbox(
                "3. Select Attribute for Grouping",
                options=metadata_df.columns,
                help="Choose the metadata column to color the points in the PCA plot."
            )

            # --- Visualization ---
            st.header("📊 PCA Results")

            # Display explained variance
            explained_variance = pca.explained_variance_ratio_
            st.markdown(f"""
            - **Principal Component 1 (PC1):** explains `{explained_variance[0]:.2%}` of the variance.
            - **Principal Component 2 (PC2):** explains `{explained_variance[1]:.2%}` of the variance.
            - **Principal Component 3 (PC3):** explains `{explained_variance[2]:.2%}` of the variance.
            - **Total Variance Explained by Top 3 PCs:** `{np.sum(explained_variance):.2%}`
            """)

            # Create tabs for Scores and Loadings plots
            tab1, tab2 = st.tabs(["Sample Scores Plot", "Feature Loadings Plot"])

            with tab1:
                st.subheader("Interactive 3D PCA Plot (Sample Scores)")
                # Create the interactive 3D scatter plot for PCA scores
                fig_scores = px.scatter_3d(
                    final_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color=grouping_attribute,
                    hover_name=final_df.index,
                    hover_data={col: True for col in metadata_df.columns},
                    title=f'3D PCA of Samples (Grouped by {grouping_attribute})',
                    labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
                    height=700
                )
                fig_scores.update_layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1)
                )
                st.plotly_chart(fig_scores, use_container_width=True)

            with tab2:
                st.subheader("Interactive 3D PCA Plot (Feature Loadings)")
                # Get the PCA loadings
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'], index=data_df.index)

                # Create the interactive 3D scatter plot for loadings
                fig_loadings = px.scatter_3d(
                    loadings_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    hover_name=loadings_df.index,
                    title='3D PCA Feature Loadings',
                    labels={'PC1': 'PC1 Loading', 'PC2': 'PC2 Loading', 'PC3': 'PC3 Loading'},
                    height=700
                )
                fig_loadings.update_traces(marker=dict(size=5, color='rgba(135, 206, 250, 0.8)', line=dict(width=1, color='DarkSlateGrey')))
                fig_loadings.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_loadings, use_container_width=True)
                st.markdown("""
                **Feature Loadings Plot:** This plot shows how much each feature influences the principal components. Features that are further from the origin have a stronger influence. Features that cluster together may be correlated.
                """)


            # --- Display DataFrames ---
            with st.expander("Show Processed Data"):
                st.subheader("PCA Scores and Metadata")
                st.dataframe(final_df)
                st.subheader("Feature Loadings")
                st.dataframe(loadings_df)
                st.subheader("Original Transposed Data")
                st.dataframe(data_df_transposed)

        except Exception as e:
            st.error(f"An error occurred during PCA processing: {e}")
            st.exception(e) # Provides a full traceback for debugging

else:
    st.info("👋 Welcome! Please upload your data and metadata files to begin the analysis.")


# --- How to Run This App ---
# 1. Save this code as a Python file (e.g., `main.py`).
# 2. Create a `requirements.txt` file with the following content:
#    streamlit
#    pandas
#    scikit-learn
#    plotly
# 3. Open your terminal and run the following commands:
#    pip install -r requirements.txt
#    streamlit run main.py

