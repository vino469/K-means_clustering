import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

st.title("🔵 K-Means Clustering App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns for clustering.")
    else:
        st.subheader("🔢 Select Features")
        selected_cols = st.multiselect(
            "Choose columns for clustering",
            numeric_cols,
            default=numeric_cols[:2]
        )

        if len(selected_cols) >= 2:
            X = df[selected_cols]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Choose number of clusters
            k = st.slider("Select number of clusters (K)", 2, 10, 3)

            if st.button("Run K-Means"):
                # Train KMeans
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                # Add cluster column
                df["Cluster"] = clusters

                # Save model & scaler
                joblib.dump(kmeans, "kmeans_model.pkl")
                joblib.dump(scaler, "scaler.pkl")

                st.success("✅ Model saved as kmeans_model.pkl")

                st.subheader("📊 Clustered Data")
                st.dataframe(df.head())

                # Plot clusters
                fig, ax = plt.subplots()
                ax.scatter(
                    X_scaled[:, 0],
                    X_scaled[:, 1],
                    c=clusters,
                    cmap="viridis",
                    alpha=0.6
                )
                ax.scatter(
                    kmeans.cluster_centers_[:, 0],
                    kmeans.cluster_centers_[:, 1],
                    marker="X",
                    s=200,
                    c="red"
                )
                ax.set_xlabel(selected_cols[0])
                ax.set_ylabel(selected_cols[1])
                ax.set_title("K-Means Clustering Result")

                st.pyplot(fig)

        else:
            st.warning("Please select at least 2 features for clustering.")
