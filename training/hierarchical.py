import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score
from util.split_sets import split_and_standardize

# ------------------------------------------------------------------

def split_sets(train_df, test_df, continuous_features, categorical_features, outcome):
    z_score_features = [f"{col}_zscore" for col in continuous_features]
    features = z_score_features + categorical_features
    
    X_train = np.asarray(train_df[features])
    X_train_swinging = np.asarray(train_df[train_df['Swinging'] == 1][features])
    X_train_nonswinging = np.asarray(train_df[train_df['Swinging'] == 0][features])
    X_test = np.asarray(test_df[features])
    
    y_train_isswinging = np.asarray(train_df[['Swinging']])
    y_train_swinging = np.asarray(train_df[train_df['Swinging'] == 1][outcome])
    y_train_nonswinging = np.asarray(train_df[train_df['Swinging'] == 0][outcome])
    y_test = np.asarray(test_df[outcome])
    
    return X_train, X_train_swinging, X_train_nonswinging, X_test, y_train_isswinging, y_train_swinging, y_train_nonswinging, y_test

# ------------------------------------------------------------------

class HierarchicalModel():
    def __init__(self, df: pd.DataFrame, continuous_features: list, categorical_features: list, outcome: list, player_code: int, player_name: str):
        df = df[df["batter"].isin([player_code])]
        df["Swinging"] = pd.to_numeric(df["Outcome"], errors="coerce").isin([2, 3]).astype(int)
        self.player_name = player_name
        
        train_df, test_df = split_and_standardize(df, continuous_features)
        X_train, X_train_swinging, X_train_nonswinging, X_test, y_train_isswinging, y_train_swinging, y_train_nonswinging, y_test = split_sets(train_df, test_df, continuous_features, categorical_features, outcome)
        
        classifier_isswinging = svm.SVC(kernel='rbf', gamma='scale', C=50, class_weight='balanced')
        classifier_isswinging.fit(X_train, y_train_isswinging)
        y_predict_isswinging = classifier_isswinging.predict(X_test)
        
        classifier_swinging = svm.SVC(kernel='rbf', gamma='scale', C=5, class_weight='balanced')
        classifier_swinging.fit(X_train_swinging, y_train_swinging)
        y_predict_swinging = classifier_swinging.predict(X_test)
        
        classifier_nonswinging = svm.SVC(kernel='rbf', gamma='scale', C=5, class_weight='balanced')
        classifier_nonswinging.fit(X_train_nonswinging, y_train_nonswinging)
        y_predict_nonswinging = classifier_nonswinging.predict(X_test)
    
        y_predict = []
        
        for i in range(len(y_predict_isswinging)):
            if y_predict_isswinging[i] == 1:
                y_predict.append(y_predict_swinging[i])
            else:
                y_predict.append(y_predict_nonswinging[i])
        
        test_df["True Label"] = y_test
        test_df["Predicted Label"] = y_predict
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_predict = y_predict
        self.train_df = train_df
        self.test_df = test_df
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
    
    # ------------------------------------------------------------------
        
    def evaluate(self):
        cm = confusion_matrix(self.y_test, self.y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(cmap='Blues')
        plt.title(f"{self.player_name} Hierarchical Model Testing")
        plt.show()

        print(classification_report(self.y_test, self.y_predict))
    
    # ------------------------------------------------------------------
        
    def get_classification_report(self):
        return classification_report(self.y_test, self.y_predict)

    # ------------------------------------------------------------------
    
    def get_summary(self):
        return {
            "Hierarchical accuracy": float(accuracy_score(self.y_test, self.y_predict)),
            "Hierarchical precision": float(precision_score(self.y_test, self.y_predict, average="macro", zero_division=0)),
            "Hierarchical recall": float(recall_score(self.y_test, self.y_predict, average="macro", zero_division=0)),
        }
  
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
    
    judge_svm = HierarchicalModel(df, continuous_features, categorical_features, outcomes, 592450, "Aaron Judge")
    
    print(judge_svm.get_summary())