import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the AdvBench Dataset
dataset = load_dataset("walledai/AdvBench")

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()

# Step 2: Split into Training and Testing Sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # 80-20 split

# Step 3: Load the Toxicity Classifier
model_name = "unitary/toxic-bert"  # Replace with RoBERTa-based model if available
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, return_all_scores=True)

# Step 4: Define a Function to Get Toxicity Scores
def compute_toxicity_scores(texts, batch_size=32):
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = classifier(batch)
        for result in batch_results:
            # Extract 'toxic' score
            toxic_score = 0.0
            for label in result:
                if label['label'].lower() == 'toxic':
                    toxic_score = label['score']
                    break
            scores.append(toxic_score)
    return scores

# Step 5: Apply the Classifier to the Testing Set
test_questions = test_df['prompt'].tolist()

print("Computing toxicity scores for the test set...")
test_toxicity_scores = compute_toxicity_scores(test_questions)
test_df = test_df.copy()  # To avoid SettingWithCopyWarning
test_df['toxicity_score'] = test_toxicity_scores

# Step 6: Select the Top 50 Most Harmful Questions
top_50 = test_df.nlargest(50, 'toxicity_score').copy()

print("Top 50 most harmful questions based on toxicity scores:")
print(top_50[['prompt', 'toxicity_score']])

top_50.reset_index(drop=True, inplace=True)

# Display the DataFrame
print("\nTop 50 Toxic Questions DataFrame:")
print(top_50[['prompt', 'toxicity_score']])

# (Optional) Step 8: Save the Top 50 DataFrame to a CSV File
output_csv_path = "top_50_toxic_questions.csv"
top_50[['prompt', 'toxicity_score']].to_csv(output_csv_path, index=False)
print(f"\nTop 50 toxic questions have been saved to '{output_csv_path}'.")


# Step 7: Visualize Toxicity Score vs. Question Index

# Reset index for plotting
test_df = test_df.reset_index(drop=True)

# Sort the test_df by toxicity score in descending order for better visualization
test_df_sorted = test_df.sort_values(by='toxicity_score', ascending=False).reset_index(drop=True)

# Assign a rank/index after sorting
test_df_sorted['rank'] = test_df_sorted.index + 1

# Create a boolean column to indicate top 50
test_df_sorted['is_top_50'] = test_df_sorted['toxicity_score'] >= test_df_sorted['toxicity_score'].nlargest(50).min()

# Plotting
plt.figure(figsize=(14, 7))
sns.scatterplot(
    data=test_df_sorted,
    x='rank',
    y='toxicity_score',
    hue='is_top_50',
    palette={True: 'red', False: 'blue'},
    alpha=0.6
)

plt.title('Toxicity Score vs. Question Index')
plt.xlabel('Question Index (Ranked by Toxicity)')
plt.ylabel('Toxicity Score')
plt.legend(title='Top 50', labels=['Top 50', 'Others'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Annotate Top 50 Points
plt.figure(figsize=(14, 7))
sns.scatterplot(
    data=test_df_sorted,
    x='rank',
    y='toxicity_score',
    hue='is_top_50',
    palette={True: 'red', False: 'blue'},
    alpha=0.6
)

# Annotate the top 50 points
for _, row in top_50.iterrows():
    # Find the rank of this row in the sorted dataframe
    rank = test_df_sorted[test_df_sorted['prompt'] == row['prompt']].index[0] + 1
    plt.text(rank, row['toxicity_score'], '*', color='black', fontsize=12)

plt.title('Toxicity Score vs. Question Index with Top 50 Highlighted')
plt.xlabel('Question Index (Ranked by Toxicity)')
plt.ylabel('Toxicity Score')
plt.legend(title='Top 50', labels=['Top 50', 'Others'])
plt.grid(True)
plt.tight_layout()
plt.show()

