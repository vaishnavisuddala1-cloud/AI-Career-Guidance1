import pandas as pd

# Step 1: Load datasets
ai_jobs = pd.read_csv("ai_job_market_insights.csv")
ai_research = pd.read_csv("arxiv_ai.csv")
ai_salaries = pd.read_csv("salaries.csv")

# Step 2: Show basic information
print("✅ AI Jobs Dataset Preview:")
print(ai_jobs.head(), "\n")

print("✅ AI Research Dataset Preview:")
print(ai_research.head(), "\n")

print("✅ AI Salaries Dataset Preview:")
print(ai_salaries.head(), "\n")

# Step 3: Save dataset summaries for later reference
with open("dataset_summary.txt", "w", encoding="utf-8") as f:
    f.write("AI Jobs Dataset Info:\n")
    f.write(str(ai_jobs.info()) + "\n\n")
    
    f.write("AI Research Dataset Info:\n")
    f.write(str(ai_research.info()) + "\n\n")
    
    f.write("AI Salaries Dataset Info:\n")
    f.write(str(ai_salaries.info()) + "\n\n")

print("📂 Dataset summaries saved in dataset_summary.txt")
