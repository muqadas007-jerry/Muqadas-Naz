import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
data = pd.read_csv("dataset.csv")
data.head()
data.dtypes
data.isna().sum()
data.columns
X = data[columns[:-1]]  # Features (all columns except 'Target')
y = data['Target']      # Target variable
# Standardize the featuresDropped_Out
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and train the SVM model
model = SVC(kernel='linear')  # You can change the kernel as needed
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
caler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
!pip install scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ... (your existing code for data loading, feature encoding, and scaling) ...

# Encode the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
# Make predictions with Random Forest Regressor
rf_predictions = rf_regressor.predict(X_test)

# Print Random Forest predictions
print("Random Forest Regressor Predictions:")
print(rf_predictions)
 #Ensemble SVM and Random Forest through Voting Classifier
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svm_model = SVC(probability=True, random_state=42)  # Enable probability for voting
voting_classifier = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_regressor)
], voting='soft')
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # Import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ... (your existing code for data loading, feature encoding, and scaling) ...

svm_model = SVC(probability=True, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42) # Use RandomForestClassifier instead of RandomForestRegressor

voting_classifier = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_classifier) # Replace rf_regressor with rf_classifier
], voting='soft')

# Train the Voting Classifier
voting_classifier.fit(X_train, y_train)
# Make predictions with Voting Classifier
voting_predictions = voting_classifier.predict(X_test)

# Evaluate the Voting Classifier
print("Voting Classifier Predictions:")
print(voting_predictions)
print(confusion_matrix(y_test, voting_predictions))
print(classification_report(y_test, voting_predictions))
