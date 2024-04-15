import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob


from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib

def get_time_from_file_name(file_path, segment_length = 2000):
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)

    segmment_time = int(''.join(filter(str.isdigit, file_name))) * segment_length
    return segmment_time


path_to_annotations = "/videos/mpi_data/2Itzik/dyadic_communication/annotations.csv"
df_raw = pd.read_csv(path_to_annotations)

df_raw['label'] = df_raw['label'].replace('shrg', 'shrai')
df_raw['tier'] = df_raw['tier'].replace('hands/arms', 'hands_arms')

combinations_df = df_raw.groupby(['tier', 'label']).size().reset_index(name='count')

feature_names = df_raw['label'].unique()
# tier_names = df_raw['tier'].unique()

# Create a new column indicating the presence of a label
df_raw['label_presence'] = 1

# Pivot the table
reshaped_df = df_raw.groupby(['begin', 'end', 'seat', 'label'])['label_presence'].max().unstack(level=['label'], fill_value=0)

# Reset index to make 'begin', 'end' as columns
reshaped_df = reshaped_df.reset_index()

# Add this line after reshaped_df = reshaped_df.reset_index()
reshaped_df.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col for col in reshaped_df.columns]

# Remove training _
reshaped_df.columns = [col[:-1] if col.endswith('_') else col for col in reshaped_df.columns]

# Remove end time and make begin the index
reshaped_df = reshaped_df.drop('end', axis=1).set_index(['begin','seat'])

#==============================================================================================================================
path_to_videos = '/videos/mpi_data/2Itzik/dyadic_communication/PIS_ID_000_FEATURES/VideoMAE_ViT-B_checkpoint_Kinetics-400/'
all_video_files = glob.glob(osp.join(path_to_videos,'*','*','*','*.npy'))
all_video_files
df_features = pd.DataFrame(all_video_files, columns=['filenames'])
df_features['view'] = ['top' if 'Cam3' in c else 'front' for c in all_video_files]
df_features['seat'] = ['left' if 'left' in c else 'front' for c in all_video_files]
df_features['begin'] = df_features['filenames'].apply(get_time_from_file_name)
df_features['features'] = df_features['filenames'].apply(lambda x: np.load(x))

#==============================================================================================================================

df_vis = df_features[['begin','seat','features']].join(reshaped_df, on=['begin','seat']).fillna(0)
df_vis['bg'] = (df_vis[feature_names] == 0).all(axis=1)


df_vis_move = df_vis.loc[~df_vis['bg']]
features = np.vstack(df_vis_move['features'].values)
labels = df_vis_move[feature_names].values

#==============================================================================================================================


X = features
y = labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a multi-label SVM classifier
svm_classifier = SVC(kernel='linear', probability=True, verbose=1)
multi_label_svm = MultiOutputClassifier(svm_classifier)

# Train the multi-label SVM classifier
multi_label_svm.fit(X_train, y_train)

# Make predictions on the test set
predictions = multi_label_svm.predict(X_test)

# Evaluate the performance (accuracy in this example)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


#==============================================================================================================================

# Assuming 'multi_label_svm', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test' are defined

# Save the MultiOutputClassifier
joblib.dump(multi_label_svm, 'multi_label_svm_model.joblib')

# Save the input features and target labels
joblib.dump(X, 'input_features.joblib')
joblib.dump(y, 'target_labels.joblib')

# Save the train/test splits
joblib.dump(X_train, 'X_train.joblib')
joblib.dump(X_test, 'X_test.joblib')
joblib.dump(y_train, 'y_train.joblib')
joblib.dump(y_test, 'y_test.joblib')

