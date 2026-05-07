import pandas as pd
import matplotlib.pyplot as plt


def main():

    # ==========================================
    # Load Dataset
    # ==========================================

    df = pd.read_csv('../data/data.csv')

    # ==========================================
    # Display Dataset Information
    # ==========================================

    print("\nFirst 5 Rows:")
    print(df.head(5))

    print("\nLast 3 Rows:")
    print(df.tail(3))

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Shape:")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {len(df.index)}")

    print("\nColumn Names:")
    print(df.columns)

    # ==========================================
    # Missing Value Analysis
    # ==========================================

    print("\nMissing Values:")
    print(df.isnull().sum())

    # ==========================================
    # Unique Value Analysis
    # ==========================================

    print("\nUnique Species IDs:")
    print(df['species_id'].unique())

    site_names = df['plot_id'].unique()

    print(f"\nNumber of Unique Sites: {len(site_names)}")
    print(f"Number of Unique Species: {len(df['species_id'].unique())}")

    # ==========================================
    # GroupBy Analysis
    # ==========================================

    print("\nWeight Summary by Site:")
    print(df.groupby('plot_id')['weight'].describe())

    print("\nSpecies Sample Counts:")
    print(df['species_id'].value_counts())

    # ==========================================
    # Data Transformation
    # ==========================================

    df['weight'] = df['weight'].apply(lambda x: x * 2)

    # ==========================================
    # Visualization 1:
    # Species Distribution
    # ==========================================

    species_count = df['species_id'].value_counts()

    plt.figure(figsize=(10, 5))
    plt.bar(species_count.index, species_count.values)

    plt.xlabel('Species')
    plt.ylabel('Number of Samples')
    plt.title('Species Distribution')

    plt.savefig('../images/species_distribution.png')
    plt.show()

    # ==========================================
    # Visualization 2:
    # Average Weight Per Site
    # ==========================================

    site_weight_mean = df.groupby('plot_id')['weight'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(site_weight_mean.index, site_weight_mean.values)

    plt.xlabel('Site')
    plt.ylabel('Average Weight')
    plt.title('Average Weight Across Sites')

    plt.savefig('../images/average_weight_per_site.png')
    plt.show()

    # ==========================================
    # Visualization 3:
    # Gender Distribution
    # ==========================================

    sex_count = df['sex'].value_counts()

    plt.figure(figsize=(6, 5))
    plt.bar(sex_count.index, sex_count.values)

    plt.xlabel('Sex')
    plt.ylabel('Number of Samples')
    plt.title('Gender Distribution Across Dataset')

    plt.savefig('../images/gender_distribution.png')
    plt.show()

    # ==========================================
    # Visualization 4:
    # Total Weight by Sex Per Site
    # ==========================================

    site_sex_weight = (
        df.groupby(['plot_id', 'sex'])['weight']
        .sum()
        .unstack()
    )

    site_sex_weight.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6)
    )

    plt.xlabel('Site')
    plt.ylabel('Total Weight')
    plt.title('Total Weight by Sex for Each Site')

    plt.savefig('../images/total_weight_by_sex_site.png')
    plt.show()


if __name__ == "__main__":
    main()