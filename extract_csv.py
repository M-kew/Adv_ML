import pandas as pd

# Read the CSV file into a DataFrame
harmful_questions_df = pd.read_csv("/Users/mkew/top_50_toxic_answers_2.csv")

# Extract the 'prompt' column as a list
prompts_list = harmful_questions_df['prompt'].tolist()

# Extract the 'answer' column as a list
answers_lists = harmful_questions_df['output'].tolist()

# Display the list
print(prompts_list)
print(answers_lists)