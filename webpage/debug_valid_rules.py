import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("data/Synthetic_Interactions.csv")
transactions = df.groupby("user_id")["course_id"].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_tf = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_tf, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Show course_ids that appear as antecedents
antecedent_course_ids = sorted(set(item for a in rules['antecedents'] for item in a))
print("\n📌 Valid course IDs to test in /rules:")
print(antecedent_course_ids[:20])
