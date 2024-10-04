from teradataml import *

from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    # Extracting feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key
    
    print(context.dataset_info.sql)

    # Load the training data from Teradata
    train_df = DataFrame.from_query(context.dataset_info.sql)
    
    print("data: ", train_df)

    print("Starting training...")

    aml = AutoClassifier(
        exclude = 'knn',
        verbose = 2,
        max_runtime_secs = 300
    )
    
    aml.fit(train_df, target_name)

    aml.leaderboard()
    
    aml.leader()

    print("Finished training")
    
    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        context=context
    )
    
    print("All done!")
