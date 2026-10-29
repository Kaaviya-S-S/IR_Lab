import numpy as np
import pandas as pd


R = pd.DataFrame([
    [1, np.nan, 2, np.nan, 3],
    [2, 5, 4, np.nan, np.nan],
    [np.nan, 1, 2, 1, np.nan],
    [5, 4, np.nan, 1, 4],
    [2, 2, 1, 2, 5]
], index=['U1', 'U2', 'U3', 'U4', 'U5'],
   columns=['I1', 'I2', 'I3', 'I4', 'I5'])

print("Rating Matrix:")
print(R)

# ---------------------------------------------------------

# User based similarities

def cosine_similarity_user(u,v,R):
    common = R.loc[u].notna() & R.loc[v].notna() # corrated
    x = R.loc[u, common]
    y = R.loc[v, common]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if len(x) > 0 and denominator > 0:
        return np.dot(x, y) / denominator
    return 0

def pearson_correlation_user(u,v,R):
    common = R.loc[u].notna() & R.loc[v].notna()
    x = R.loc[u, common]
    y = R.loc[v, common]
    if len(x) < 2:
       return 0
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator > 0:
        return np.dot(x, y) / denominator
    return 0

# ---------------------------------------------------------

# Item based similarities

def cosine_similarity_item(i, j, R):
    common = R[i].notna() & R[j].notna()
    x = R.loc[common, i]
    y = R.loc[common, j]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if len(x) > 0 and denominator > 0:
        return np.dot(x, y) / denominator
    return 0


def pearson_similarity_item(i, j, R):
    common = R[i].notna() & R[j].notna()
    x = R.loc[common, i]
    y = R.loc[common, j]
    if len(x) < 2:
        return 0
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator > 0:
        return np.dot(x, y) / denominator
    return 0


def adjusted_cosine_similarity_item(i, j, R):
    common = R[i].notna() & R[j].notna()
    x = R.loc[common, i]
    y = R.loc[common, j]
    user_avg = R.mean(axis=1)     # Average rating of each user
    # Subtract each common user's average rating
    x = x - user_avg.loc[common]
    y = y - user_avg.loc[common]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if len(x) > 0 and denominator > 0:
        return np.dot(x, y) / denominator
    return 0

# --------------------------------------------------------

def user_similarity(R):
    users = R.index
    S = pd.DataFrame(0.0, index=users, columns=users)
    for u in users:
        for v in users:
            S.loc[u, v] = cosine_similarity_user(u, v, R)  # use which ever similarity is needed
    return S

def item_similarity(R):
    items = R.columns
    S = pd.DataFrame(0.0, index=items, columns=items)
    for i in items:
        for j in items:
            S.loc[i, j] = adjusted_cosine_similarity_item(i, j, R) # use which ever similarity is needed
    return S

# --------------------------------------------------------

# Prediction

def predict_rating(R, S, user, item, k=None):
    # Find the top k neighbours first
    neighbours = S.loc[user].drop(user)
    neighbours = neighbours.sort_values(ascending=False).head(k)
    
    # Remove neighbours who haven't rated the target item
    neighbours = neighbours[R.loc[neighbours.index, item].notna()]

    numerator = 0
    denominator = 0
    for v, sim in neighbours.items():
        numerator += sim * R.loc[v, item]
        denominator += sim
    if denominator == 0:
        return np.nan

    pred = numerator / denominator 
    return np.round(pred)       



def predict(R,S,user,item,k=None):
    neighbours = R[item].drop(item)
    neighbours = neighbours.sort_values(ascending=False).head(k)
    
    neighbours = neighbours[R[user,neighbours.index].notna()]
    
    numerator = 0
    denominator = 0
    for j, sim in neighbours.items():
        numerator += sim * R[user, j]
        denominator += sim
    if denominator == 0:
        return np.nan

    pred = numerator / denominator 
    return np.round(pred)       
    
    
# --------------------------------------------------------

# Error analysis

# for user based. 
# similarly adapt for content based --> item_similarity

def error_analysis(R):
    actual, predicted = [], []

    for u in R.index:
        for i in R.columns:
            if pd.isna(R.loc[u, i]):
                continue

            test = R.copy()
            a = test.loc[u, i]
            test.loc[u, i] = np.nan

            S_test = user_similarity(test)
            p = predict_rating(test, S_test, u, i, k=3)

            if pd.notna(p):
                actual.append(a)
                predicted.append(p)


    if len(actual) == 0:
        print("No valid predictions available.")
        return
    
    # Calculate MAE and MSE
    errors = np.array(actual) - np.array(predicted)
    MAE = np.mean(np.abs(errors))
    MSE = np.mean(errors ** 2)

    print("Number of predictions:", len(actual))
    print("MAE:", MAE)
    print("MSE:", MSE)
    print("Actual: ", np.array(actual).tolist())
    print("predicted: ", np.array(predicted).tolist())


# --------------------------------------------------------

S1 = user_similarity(R)
print(S1)

S2 = item_similarity(R)
print(S2)

# top 3 neighbours of I4
top3_items = S2.loc['I4'].drop('I4').sort_values(ascending=False).head(3)
# print(top3_items)

# top 3 neighbours of U1
top3_users = S1.loc['U1'].drop('U1').sort_values(ascending=False).head(3)
# print(top3_users)

# predicting U1,I4 using top 3 neighbours of U1
res = predict_rating(R, S1, 'U1', 'I4', 3)
print(res)

error_analysis(R)