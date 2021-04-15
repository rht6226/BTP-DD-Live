from helpers import get_dataset, print_analysis, average
from sklearn.neighbors import KNeighborsClassifier
import joblib

(X_train, y_train), (X_test, y_test) = get_dataset()

# Define KNN and Fit data
neigh = KNeighborsClassifier(n_neighbors=80)
neigh.fit(X_train, y_train)

# Predict values 
y_pred = neigh.predict(X_test)
y_pred = average(y_pred)

# Get Score
y_score = neigh.predict_proba(X_test)[:,1]


print_analysis(y_test, y_pred, y_score)


# save the model
joblib.dump(neigh, './models/classifier.pkl')


