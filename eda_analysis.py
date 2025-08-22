import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
ai_jobs = pd.read_csv(r"C:\Users\vaish\OneDrive\careerguidance\AI-Future-Trends\data\ai_job_market_insights.csv")
ai_research = pd.read_csv(r"C:\Users\vaish\OneDrive\careerguidance\AI-Future-Trends\data\arxiv_ai.csv")
ai_salaries = pd.read_csv(r"C:\Users\vaish\OneDrive\careerguidance\AI-Future-Trends\data\salaries.csv")

# -------------------
# 1. Basic Cleaning
# -------------------
print("Missing values in Jobs Dataset:\n", ai_jobs.isnull().sum(), "\n")
print("Missing values in Research Dataset:\n", ai_research.isnull().sum(), "\n")
print("Missing values in Salaries Dataset:\n", ai_salaries.isnull().sum(), "\n")

# Drop duplicates
ai_jobs = ai_jobs.drop_duplicates()
ai_research = ai_research.drop_duplicates()
ai_salaries = ai_salaries.drop_duplicates()

# -------------------
# 2. Jobs Insights
# -------------------
top_jobs = ai_jobs['Job_Title'].value_counts().head(10)
print("Top AI Jobs:\n", top_jobs)

plt.figure(figsize=(8,5))
sns.barplot(x=top_jobs.values, y=top_jobs.index, hue=top_jobs.index, palette="viridis", legend=False)

plt.title("Top AI Job Titles")
plt.xlabel("Count")
plt.ylabel("Job Title")
plt.tight_layout()
plt.savefig("top_ai_jobs.png")
plt.show()

# -------------------
# 3. Salary Insights
# -------------------
salary_trends = ai_salaries.groupby("work_year")["salary_in_usd"].mean()
print("\nAverage Salary Trend:\n", salary_trends)

plt.figure(figsize=(8,5))
salary_trends.plot(kind="line", marker="o")
plt.title("AI Salaries Over Time (USD)")
plt.xlabel("Year")
plt.ylabel("Average Salary (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ai_salary_trends.png")
plt.show()

# -------------------
# 4. Research Insights
# -------------------
research_trends = ai_research['primary_category'].value_counts().head(10)
print("\nTop Research Categories:\n", research_trends)

plt.figure(figsize=(8,5))
sns.barplot(x=research_trends.values, y=research_trends.index, hue=research_trends.index, palette="magma", legend=False)
plt.title("Top AI Research Categories")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("top_research_categories.png")
plt.show()
