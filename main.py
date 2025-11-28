import pandas as pd



def main():
    df = pd.read_csv("./AI Tools Adoption.csv")

    print(df.info())
    print(df.head(30))

    # See how your categories are distributed.
    categorical_cols = ['user_id', 'role', 'education_level', 'subject_area', 
                    'ai_tool', 'ai_use_purpose', 'sentiment_label', 'emotion_primary']

    for col in categorical_cols:
        print(df[col].value_counts())


if __name__ == "__main__":
    main()
