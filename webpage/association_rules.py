import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Load rules from transaction data
def load_rules():
    df = pd.read_csv("data/Synthetic_Interactions.csv")
    transactions = df.groupby("user_id")["course_id"].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_tf, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    print("\n[DEBUG] 🔍 Rule Antecedents:")
    for a in rules['antecedents']:
        print(list(a))
    
    return rules


# Step 2: Filter by selected course_id
def explain_rules_for_course(course_id):
    rules = load_rules()
    filtered = rules[rules['antecedents'].apply(lambda x: course_id in list(x))]

    explanations = []
    for _, row in filtered.iterrows():
        consequent = list(row['consequents'])[0]
        explanations.append({
            "consequent": consequent,
            "support": round(row['support'], 2),
            "confidence": round(row['confidence'], 2),
            "lift": round(row['lift'], 2)
        })

    return explanations
