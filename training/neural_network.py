import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score
from util.split_sets import split_and_standardize

# ------------------------------------------------------------------

def split_sets(train_df, test_df, continuous_features, categorical_features, outcome):
    z_score_features = [f"{col}_zscore" for col in continuous_features]
    
    X_cont_train = train_df[z_score_features].apply(pd.to_numeric, errors="coerce")
    X_cat_train = train_df[categorical_features].astype(int)
    X_cont_test = test_df[z_score_features].apply(pd.to_numeric, errors="coerce")
    X_cat_test = test_df[categorical_features]
    y_train = train_df[outcome]
    y_test = test_df[outcome]
    
    return X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test

# ------------------------------------------------------------------

# Based on sample code provided by ChatGPT
# Prompt asked about how to make a neural network with both continuous and categorical features
class MixedNN(nn.Module):
    def __init__(self, num_continuous, cat_cardinalities, hidden_size, num_classes):
        super().__init__()
        # Embeddings for categorical columns
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality+1)//2))  # simple rule of thumb
            for cardinality in cat_cardinalities
        ])
        total_emb_size = sum(e.embedding_dim for e in self.embeddings)
        input_size = num_continuous + total_emb_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x_cont, x_cat):
        # Embed each categorical column, concatenate with continuous inputs
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat(emb + [x_cont], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# ------------------------------------------------------------------

# Based on sample code provided by ChatGPT
# Prompt asked about providing sample code for training a neural network
class NeuralNetworkModel():
    def __init__(self, df: pd.DataFrame, continuous_features: list, categorical_features: list, outcome: list, player_code: int, player_name: str):
        df = df[df["batter"].isin([player_code])]
        self.player_name = player_name
        
        
        train_df, test_df = split_and_standardize(df, continuous_features)

        # Encode Categorical Columns 
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            label_encoders[col] = le

        # Compute cardinalities BEFORE converting to tensor
        cat_cardinalities = [len(label_encoders[col].classes_) for col in categorical_features]

        X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test = split_sets(train_df, test_df, continuous_features, categorical_features, outcome)

        # Convert to tensors
        X_train_cont = torch.tensor(X_cont_train.values, dtype=torch.float32)
        X_test_cont = torch.tensor(X_cont_test.values, dtype=torch.float32)
        X_train_cat = torch.tensor(X_cat_train.values, dtype=torch.long)
        X_test_cat = torch.tensor(X_cat_test.values, dtype=torch.long)
        Y_train = torch.tensor(y_train.values, dtype=torch.long).squeeze()
        Y_test = torch.tensor(y_test.values, dtype=torch.long).squeeze()

        # Instantiate Model
        model = MixedNN(
            num_continuous=len(continuous_features),
            cat_cardinalities=cat_cardinalities,
            hidden_size=128,
            num_classes=4
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_cont, X_train_cat)
            
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # Evaluate
        with torch.no_grad():
            y_predict = torch.argmax(model(X_test_cont, X_test_cat), dim=1)
            acc = (y_predict == Y_test).float().mean()
            print(f"Test Accuracy: {acc:.4f}")
        
        test_df["True Label"] = y_test
        test_df["Predicted Label"] = y_predict
         
        self.X_train_cont = X_train_cont
        self.X_test_cont = X_test_cont
        self.X_train_cat = X_train_cat
        self.X_test_cat = X_test_cat
        self.y_train = y_train
        self.y_test = y_test
        self.y_predict = y_predict
        self.train_df = train_df
        self.test_df = test_df
        self.model = model
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        
    # ------------------------------------------------------------------  
        
    def evaluate(self):
        cm = confusion_matrix(self.y_test, self.y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(cmap='Blues')
        plt.title(f"{self.player_name} Neural Network Model Testing")
        plt.show()

        print(classification_report(self.y_test, self.y_predict))
        
    # ------------------------------------------------------------------  
        
    def get_classification_report(self):
        return classification_report(self.y_test, self.y_predict)
    
    # ------------------------------------------------------------------  
    
    def get_summary(self):
        return {
            "NN accuracy": float(accuracy_score(self.y_test, self.y_predict)),
            "NN precision": float(precision_score(self.y_test, self.y_predict, average="macro", zero_division=0)),
            "NN recall": float(recall_score(self.y_test, self.y_predict, average="macro", zero_division=0)),
        }
    
    # ------------------------------------------------------------------  
    
    # Citation: ChatGPT helped by providing example code for this 
    # Prompt asked about how to find feature importance from a neural network model
    def get_feature_importance(self):
        feature_names = self.continuous_features + self.categorical_features
        
        def predict_fn(X_cont, X_cat):
            self.model.eval()
            with torch.no_grad():
                logits = self.model(
                    torch.tensor(X_cont, dtype=torch.float32),
                    torch.tensor(X_cat, dtype=torch.long)
                )
                return logits.argmax(dim=1).numpy()

        self.model.eval()
        # Baseline predictions
        baseline = predict_fn(self.X_train_cont, self.X_train_cat)
        baseline_score = accuracy_score(self.y_train, baseline)

        importances = []

        # Combine continuous & categorical features for easier permutation
        X = np.hstack([self.X_train_cont, self.X_train_cat])
        num_features = X.shape[1]

        # For mapping back to cont/cat split
        n_cont = self.X_train_cont.shape[1]

        for col in range(num_features):
            scores = []
            for _ in range(5):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, col])

                # Split back
                Xc_perm = X_permuted[:, :n_cont]
                Xcat_perm = X_permuted[:, n_cont:].astype(int)

                preds = predict_fn(Xc_perm, Xcat_perm)
                score = accuracy_score(self.y_train, preds)
                scores.append(baseline_score - score)

            importances.append(np.mean(scores))

        # Create final dictionary
        importance_dict = {
            feature_names[i]: float(importances[i])
            for i in range(num_features)
        }

        return importance_dict
  
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
    
    judge_svm = NeuralNetworkModel(df, continuous_features, categorical_features, outcomes, 592450, "Aaron Judge")
    
    print(judge_svm.get_summary())
    
    print(judge_svm.get_feature_importance())