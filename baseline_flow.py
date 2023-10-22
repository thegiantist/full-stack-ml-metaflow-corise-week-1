from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
)
from metaflow.cards import Table, Markdown, Artifact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Labeling function
def labeling_function(row):
    if row['rating'] >= 4:
        return 1
    else:
        return 0

# Function to train and evaluate a baseline model
def train_and_evaluate_baseline(train_data, val_data):
    # Splitting the data into features and labels
    X_train, y_train = train_data['review'], train_data['label']
    X_val, y_val = val_data['review'], val_data['label']
    
    # Converting the reviews into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_val_transformed = vectorizer.transform(X_val)
    
    # Training a Logistic Regression model
    classifier = LogisticRegression()
    classifier.fit(X_train_transformed, y_train)
    
    # Making predictions on the validation data
    y_pred = classifier.predict(X_val_transformed)
    
    # Calculating the accuracy and ROC AUC score
    acc = accuracy_score(y_val, y_pred)
    rocauc = roc_auc_score(y_val, y_pred)
    
    return acc, rocauc, y_pred, y_val

class BaselineNLPFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # Load dataset packaged with the flow.
        df = pd.read_csv(io.StringIO(self.data))

         # Filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)

        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # Split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        # Training the baseline model and getting predictions
        self.base_acc, self.base_rocauc, y_pred, y_val = train_and_evaluate_baseline(self.traindf, self.valdf)
        
        # Computing false positives and false negatives
        false_positives = self.valdf[(y_pred == 1) & (y_val == 0)]
        false_negatives = self.valdf[(y_pred == 0) & (y_val == 1)]
        
        # Storing the false positives and negatives for the next step
        self.false_positives = false_positives
        self.false_negatives = false_negatives

        self.next(self.end)

    @card(type="corise")
    @step
    def end(self):
        # ... [Rest of the code remains unchanged]
        
        current.card.append(Markdown("## Examples of False Positives"))
        current.card.append(Table(self.false_positives))
        
        current.card.append(Markdown("## Examples of False Negatives"))
        current.card.append(Table(self.false_negatives))

if __name__ == "__main__":
    BaselineNLPFlow()
