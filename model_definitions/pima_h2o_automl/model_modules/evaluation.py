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
    # Determine the home directory of the current user
    user_home_dir = os.path.expanduser('~')

    # Check if the 'JupyterLabRoot' directory exists
    jupyterlab_root_dir = os.path.join(user_home_dir, 'JupyterLabRoot')

    # Determine Java installation path based on the existence of 'JupyterLabRoot'
    if os.path.isdir(jupyterlab_root_dir):
        java_home_path = os.path.join(jupyterlab_root_dir, '.jdk', 'jdk-17.0.9+9')
    else:
        java_home_path = os.path.join(user_home_dir, '.jdk', 'jdk-17.0.9+9')

    # Set JAVA_HOME to the default or existing environment value
    os.environ['JAVA_HOME'] = os.getenv('JAVA_HOME', java_home_path)

    # Check if Java is already installed
    if not os.path.isdir(os.environ['JAVA_HOME']):
        print('Installing Java...')

        # Install Java in the determined directory
        jdk_install_path = jupyterlab_root_dir if os.path.isdir(jupyterlab_root_dir) else user_home_dir
        jdk.install('17', path=os.path.join(jdk_install_path, '.jdk'))

        # Update JAVA_HOME and PATH after successful installation
        os.environ['JAVA_HOME'] = java_home_path
        os.environ['PATH'] = f"{os.environ.get('PATH')}:{os.environ.get('JAVA_HOME')}/bin"

        print(f"Java installed at {os.environ['JAVA_HOME']}")
    else:
        print(f"Java is installed at {os.environ['JAVA_HOME']}")


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
