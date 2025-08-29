# Detecting-Fraud-in-Hawaii-Airbnb-Listings

This project aims to identify fraudulent Airbnb listings in Hawaii using unsupervised learning and text mining techniques. By leveraging real-world Airbnb data, the solution provides a data-driven pipeline to help the platform proactively detect and flag potentially fake or deceptive listings.

**Project Overview:**
With the rise of online rental platforms, fraudulent listings pose a significant risk to guests, hosts, and the platform’s reputation. This repository implements a set of machine learning models to detect unusual or suspicious patterns in Hawaii Airbnb listings, improving trust and safety on the platform.

**Methods:**
**Data Source:** Detailed, regularly updated data (over 35,000 listings, 75 features) scraped from Inside Airbnb for Hawaii.
**Dataset available here:** https://insideairbnb.com/get-the-data/.

**Data Preprocessing:** Handling of missing values, feature selection, encoding of categorical data, and transformation of text fields for analysis.

**Dimensionality Reduction:** Principal Component Analysis (PCA) is used to capture key variance within the dataset, improving efficiency for downstream models.

**Clustering & Outlier Detection:**

**Isolation Forests:** Finds anomalies based on their feature values; best silhouette score: 0.796 with PCA-transformed data.

**K-Means Clustering:** Groups listings into similar clusters; best performance with k=2 clusters (silhouette score: 0.85 with PCA).

**DBSCAN:** Detects density-based clusters; performed best after PCA (silhouette score: 0.82).

**Text Mining:** Uses TF-IDF and cosine similarity to flag listings with highly similar descriptions, highlighting cases of leasing the same property under different hosts or duplicate content.

**Key Findings:**
Listings flagged as suspicious tend to have fewer reviews, lower prices, unverified hosts, and reduced host engagement.
Discoveries include listings with identical descriptions but different hosts, indicating possible duplicates or scams.
The combination of anomaly detection and text analysis surfaced 236 listings flagged by all three models for further manual review.

**Limitations:**
Lack of ground-truth labels: Model flags listings based on anomalies, which require manual validation for confirmation.
Some variables exhibit significant skew, impacting feature reduction and clustering performance.

**Future Work:**
Manual and real-world validation of flagged listings.
Develop explainable model outputs for platform integration.
Expand model to other regions and datasets for broader impact.

**Getting Started:**
Download data from Inside Airbnb.
**Install dependencies:** scikit-learn, numpy, pandas, and related libraries.
Run preprocessing and modeling scripts as provided in the repository.
Analyze flagged results for further investigation.

**Stakeholders:**
**Airbnb Managers:** Direct users for daily fraud monitoring.
**Guests/Hosts:** Indirect beneficiaries through increased platform safety.
