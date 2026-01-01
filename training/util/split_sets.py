from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Based on example code provided by ChatGPT
# Prompt asked about how to calculate the Z-score on the training set
# and use that to transform the testing set specific to a certain pitch type
def split_and_standardize(df, standardize_cols):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare columns for scaled values
    for col in standardize_cols:
        train_df[f"{col}_zscore"] = None
        test_df[f"{col}_zscore"] = None

    # Dictionary to store fitted scalers per pitch type
    scalers = {}

    # --- FIT on train data ---
    for pitch_type, group in train_df.groupby("pitch_name"):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(group[standardize_cols])
        train_df.loc[group.index, [f"{col}_zscore" for col in standardize_cols]] = scaled
        scalers[pitch_type] = scaler  # save the fitted scaler

    # --- TRANSFORM test data using corresponding pitch-type scaler ---
    for pitch_type, group in test_df.groupby("pitch_name"):
        if pitch_type in scalers:  # only transform if seen during training
            scaler = scalers[pitch_type]
            scaled = scaler.transform(group[standardize_cols])
            test_df.loc[group.index, [f"{col}_zscore" for col in standardize_cols]] = scaled
        else:
            # Handle unseen pitch types — optional (could skip, or use global scaler)
            print(f"Unseen pitch type in test set: {pitch_type}")
            
    return train_df, test_df