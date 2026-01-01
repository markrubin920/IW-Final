import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score
from util.split_sets import split_and_standardize

# ------------------------------------------------------------------

def split_sets(train_df, test_df, continuous_features, categorical_features, outcome):
    z_score_features = [f"{col}_zscore" for col in continuous_features]
    features = z_score_features + categorical_features
    
    X_train = np.asarray(train_df[features])
    X_test = np.asarray(test_df[features])
    y_train = np.asarray(train_df[outcome])
    y_test = np.asarray(test_df[outcome])
    
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------

class DecisionTreeModel():
    def __init__(self, df: pd.DataFrame, continuous_features: list, categorical_features: list, outcome: list, player_code: int, player_name: str):
        df = df[df["batter"].isin([player_code])]
        self.player_name = player_name
        
        train_df, test_df = split_and_standardize(df, continuous_features)
        X_train, X_test, y_train, y_test = split_sets(train_df, test_df, continuous_features, categorical_features, outcome)
        
        classifier = DecisionTreeClassifier(
            class_weight= 'balanced', criterion= 'gini', max_depth=5,
            max_features=None, min_samples_leaf=1, min_samples_split=10, 
            splitter='random'
        )
        
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        
        test_df["True Label"] = y_test
        test_df["Predicted Label"] = y_predict
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_predict = y_predict
        self.train_df = train_df
        self.test_df = test_df
        self.classifier = classifier
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
    
    # ------------------------------------------------------------------
        
    def evaluate(self):
        cm = confusion_matrix(self.y_test, self.y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(cmap='Blues')
        plt.title(f"{self.player_name} SVM Model Testing")
        plt.show()

        print(classification_report(self.y_test, self.y_predict))
        
        plot_tree(
            self.classifier,
            feature_names=list(self.continuous_features + self.categorical_features),
            class_names=['0', '1', '2', '3'],
            filled=True,
            rounded=True
        )
    
    # ------------------------------------------------------------------
        
    def get_classification_report(self):
        return classification_report(self.y_test, self.y_predict)

    # ------------------------------------------------------------------
    
    def get_summary(self):
        return {
            "Decision Tree accuracy": float(accuracy_score(self.y_test, self.y_predict)),
            "Decision Tree precision": float(precision_score(self.y_test, self.y_predict, average="macro", zero_division=0)),
            "Decision Tree recall": float(recall_score(self.y_test, self.y_predict, average="macro", zero_division=0)),
        }
        
    # ------------------------------------------------------------------
    
    # Citation: ChatGPT helped by providing example code for this 
    # Prompt asked about how to find feature importance in a model
    def get_feature_importance(self):
        X_sub, _, y_sub, _ = train_test_split(self.X_train, self.y_train, test_size=0.6, stratify=self.y_train)
        results = permutation_importance(self.classifier, X_sub, y_sub, n_repeats=3, n_jobs=-1)
        importance = results.importances_mean.flatten()

        importance_df = pd.DataFrame({
            'feature': self.continuous_features + self.categorical_features,
            'importance': importance
        })

        return importance_df
  
# ------------------------------------------------------------------  
      
if __name__ == "__main__":
    
    df = pd.read_csv("../data/cleaning_3/clean.csv")
    
    categorical_features = [
        'zone_1.0', 'zone_2.0', 'zone_3.0', 'zone_4.0', 'zone_5.0', 'zone_6.0',
        'zone_7.0', 'zone_8.0', 'zone_9.0', 'zone_11.0', 'zone_12.0',
        'zone_13.0', 'zone_14.0',
        'pitch_name_4-Seam Fastball', 'pitch_name_Changeup',
        'pitch_name_Curveball', 'pitch_name_Cutter', 'pitch_name_Knuckle Curve',
        'pitch_name_Sinker', 'pitch_name_Slider', 'pitch_name_Split-Finger',
        'pitch_name_Sweeper', 
        'Pitcher Side'
    ]

    continuous_features = [
        'balls', 'strikes', 'outs_when_up', 
        'release_speed', 'release_spin_rate', 
        'release_extension',
        'release_pos_y', 
        'spin_axis', 
        'api_break_z_with_gravity',
        'api_break_x_arm', 
        'api_break_x_batter_in',
        'release_pos_x','release_pos_z', 
        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']

    outcomes = ['Outcome']
    
    judge_svm = DecisionTreeModel(df, continuous_features, categorical_features, outcomes, 592450, "Aaron Judge")
    
    print(judge_svm.get_summary())
    
    print(judge_svm.get_feature_importance())