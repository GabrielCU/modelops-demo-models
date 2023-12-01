from sklearn import metrics
from teradataml import DataFrame, copy_to_sql, get_context, H2OPredict
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    store_byom_tmp,
    ModelContext
)
import json
import os
import matplotlib


def check_java():
    # Set a default JAVA_HOME if it's not already set
    default_java_home = '/home/jovyan/.jdk/jdk-17.0.9+9'
    os.environ['JAVA_HOME'] = os.getenv('JAVA_HOME', default_java_home)

    # Check if Java is already installed
    if not os.path.isdir(os.environ['JAVA_HOME']):

        import jdk

        print('Installing Java...')

        jdk.install('17', path='/home/jovyan/.jdk')
        # Update JAVA_HOME and PATH after successful installation
        os.environ['JAVA_HOME'] = f'/home/jovyan/.jdk/jdk-17.0.9+9'
        os.environ['PATH'] = f"{os.environ.get('PATH')}:{os.environ.get('JAVA_HOME')}/bin"


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    with open(f"{context.artifact_input_path}/model.h2o", "rb") as f:
        model_bytes = f.read()

    model = store_byom_tmp(get_context(), "byom_models_tmp", context.model_version, model_bytes)

    target_name = context.dataset_info.target_names[0]

    byom_target_sql = "CAST(prediction AS INT)"

    check_java()

    h2o = H2OPredict(
        modeldata=model,
        newdata=DataFrame.from_query(context.dataset_info.sql),
        accumulate=[context.dataset_info.entity_key, target_name])

    predictions_df = h2o.result

    predictions_df.to_sql(table_name="predictions_tmp", if_exists="replace", temporary=True)

    metrics_df = DataFrame.from_query(f"""
    SELECT 
        {target_name} as y_test, 
        {byom_target_sql} as y_pred
        FROM predictions_tmp
    """)
    metrics_df = metrics_df.to_pandas()

    y_pred = metrics_df[["y_pred"]]
    y_test = metrics_df[["y_test"]]

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    cf = metrics.confusion_matrix(y_test, y_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cf)
    display.plot()
    save_plot('Confusion Matrix', context=context)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=context.model_version)
    display.plot()
    save_plot('ROC Curve', context=context)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(features_df=DataFrame.from_query(context.dataset_info.sql),
                                predicted_df=DataFrame("predictions_tmp"),
                                context=context)
