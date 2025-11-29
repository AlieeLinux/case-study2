import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def cat_dist():
   df = pd.read_csv("./AI Tools Adoption.csv")
      #   See how your categories are distributed.
   categorical_cols = ['user_id', 'role', 'education_level', 'subject_area', 
                    'ai_tool', 'ai_use_purpose', 'sentiment_label', 'emotion_primary']

   for col in categorical_cols:
      print(df[col].value_counts())

def bar_tool():
   df = pd.read_csv("./AI Tools Adoption.csv")

   # Creates bar graph with unique color
   sns.countplot(data=df, x='ai_tool', palette='tab20')
   plt.xticks(rotation=50)
   plt.show()


# cross tabulation
def crosstabb():
   df = pd.read_csv("./AI Tools Adoption.csv")

   # Creates bar graph with unique color
   x = pd.crosstab(df['ai_tool'], df['ai_use_purpose'])
   print(x)

# Education bar
def edu_bar():
   df = pd.read_csv("./AI Tools Adoption.csv")

   # Creates bar graph with unique color
   sns.countplot(data=df, x='education_level', palette='tab20')
   plt.xticks(rotation=45)
   plt.show()

# sentiment_label bar
def sentiment_bar():
   df = pd.read_csv("./AI Tools Adoption.csv")

   # Creates bar graph with unique color
   sns.countplot(data=df, x='sentiment_label', palette='tab20')
   plt.xticks(rotation=45)
   plt.show()

def bar_ai_use_purpose():
   df = pd.read_csv("./AI Tools Adoption.csv")

   # Creates bar graph with unique color
   sns.countplot(data=df, x='ai_use_purpose', palette='tab20')
   plt.xticks(rotation=45)
   plt.show()
# category vs numeric
def cat_vs_num():
   df = pd.read_csv("./AI Tools Adoption.csv")

   sns.boxplot(x='ai_tool', y='post_test_score', data=df)
   plt.xticks(rotation=45)
   plt.show()

def role_bar():
   df = pd.read_csv("./AI Tools Adoption.csv")

   sns.boxplot(x='role', y='post_test_score', data=df)
   plt.xticks(rotation=45)
   plt.show()

def ai_use_per_week():
   df = pd.read_csv("./AI Tools Adoption.csv")

   # Fixes 8 days
   order = sorted(df['ai_use_frequency_per_week'].unique())
   plt.figure(figsize=(8,5))
   sns.countplot(x='ai_use_frequency_per_week', data=df, palette='magma', order=order)
   plt.title("AI Use Frequency per Week")
   plt.xlabel("Days per Week")
   plt.ylabel("Number of Users")
   plt.show()


def weekly_usage():
   df = pd.read_csv("./AI Tools Adoption.csv")

   counts = df['avg_session_duration_min'].value_counts().sort_index()

   x = list(range(len(counts)))  # positions for bars/line
   y = counts.values
   
   ax = sns.barplot(x=x, y=y, palette='magma')

   sns.lineplot(x=x, y=y, color='black', marker='o', ax=ax)

   plt.title("Distribution of Average Session Duration (min)")
   plt.xlabel("Average Session Duration (minutes)")
   plt.ylabel("Number of Users")

   plt.show()

def total_weekly():
   df = pd.read_csv("./AI Tools Adoption.csv")

   sns.countplot(x='ai_use_frequency_per_week', data=df, color='skyblue')
   plt.title("Total Weekly AI Usage (Per day))")
   plt.xlabel("Days")
   plt.ylabel("Number of Users")
   plt.show()


def ethical_concern_plagiarism_bar():
   df = pd.read_csv("./AI Tools Adoption.csv")

   sns.countplot(x='ethical_concern_plagiarism', data=df, color='skyblue')
   plt.title("Ethical concerns")
   plt.xlabel("Days")
   plt.ylabel("Number of Users")
   plt.show()

def colMatrix():
   df = pd.read_csv("./AI Tools Adoption.csv")
   numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

   corr = df[numeric_cols].corr()

   plt.figure(figsize=(12,8))
   sns.heatmap(corr, annot=False, cmap='coolwarm')
   plt.show()

colMatrix()