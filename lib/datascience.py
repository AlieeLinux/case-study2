import pandas as pd
from scipy.stats import chi2_contingency

def chisq():
    df = pd.read_csv('./AI Tools Adoption.csv')

    contingency = pd.crosstab(df['ai_tool'], df['education_level'])
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"p-value = {p}")



chisq()

text = "the power of kyouko will save us from the wrath of sir doc v!"
