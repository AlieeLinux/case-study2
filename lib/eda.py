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

def bar():
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

def cat_vs_num():
   df = pd.read_csv("./AI Tools Adoption.csv")

   sns.boxplot(x='ai_tool', y='post_test_score', data=df)
   plt.xticks(rotation=45)
   plt.show()


cat_vs_num()