import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Load association rules from synthetic interaction data
def load_rules():
    # Read transaction data
    df = pd.read_csv("data/Synthetic_Interactions.csv")
    
    # Group course_ids by user_id to form transactions
    transactions = df.groupby("user_id")["course_id"].apply(list).tolist()

    # One-hot encode the transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    # Generate frequent itemsets using Apriori algorithm
    frequent_itemsets = apriori(df_tf, min_support=0.01, use_colnames=True)

    # Generate association rules with minimum confidence threshold
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    # Print rule antecedents for debugging
    print("\n[DEBUG] 🔍 Rule Antecedents:")
    for a in rules['antecedents']:
        print(list(a))
    
    return rules

# Step 2: Filter and explain rules that match a specific course_id
def explain_rules_for_course(course_id):
    # Load all rules
    rules = load_rules()

    # Filter rules where the course_id appears in the antecedents
    filtered = rules[rules['antecedents'].apply(lambda x: course_id in list(x))]

    # Prepare simplified explanation output
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
