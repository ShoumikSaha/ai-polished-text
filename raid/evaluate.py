import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def load_detection_result(df, results):
    # Load the dataframe and read in the scores
    scores_df = pd.DataFrame.from_records(results)

    # If df has a pre-existing score column, remove it before merging
    if "score" in df.columns:
        df = df.drop(columns=["score"])

    # Merge dataframes based on the id and validate that ids are unique
    return df.join(scores_df.set_index("id"), on="id", validate="one_to_one")


def compute_fpr(y_scores, threshold):
    y_pred = [1 if y >= threshold else 0 for y in y_scores]
    y_true = [0] * len(y_pred)
    return 1 - accuracy_score(y_true, y_pred)


# Search to find threshold for FPR
def find_threshold(df, target_fpr, epsilon):
    iteration = 1
    prev_dist = None
    step_size = 0.5
    y_scores = df[(df["model"] == "human") & (df["attack"] == "none") & (df["score"].notnull())]["score"].tolist()
    sign = lambda x: -1 if x < 0 else 1

    try:
        threshold = sum(y_scores) / len(y_scores)  # initialize threshold to mean of y_scores
    except ZeroDivisionError:
        raise ValueError(
            "Predictions are missing outputs for human-written texts in some domains.\n"
            + "In order to run evaluation, you must include predictions for human-written data in all domains.\n"
            + "To disable this, set per_domain_tuning=False in run_evaluation."
        ) from None

    # Initialize the list of all found thresholds and FPRs
    found_threshold_list = []
    while abs((fpr := compute_fpr(y_scores, threshold)) - target_fpr) > epsilon:
        # Save the computed values to the found_threshold_list
        found_threshold_list.append((threshold, fpr))

        # Increment the iteration count and compute distance
        iteration += 1
        dist = target_fpr - fpr

        # If dist and prev_dist are different signs then swap
        # sign of step size and cut in half
        if prev_dist and sign(dist) != sign(prev_dist):
            step_size *= -0.5
        # Otherwise if we're going the wrong direction, then just swap sign of step
        elif prev_dist and abs(dist) - abs(prev_dist) > 0.01:
            step_size *= -1

        # Step the threshold value and save prev_dist
        threshold += step_size
        prev_dist = target_fpr - fpr

        # Can't find the threshold, let's find the best one
        if iteration > 50:
            # Compute diffs for all thresholds found during search
            # (Exclude all thresholds for which the true fpr is 0)
            diffs = [(target_fpr - fpr, t) for t, fpr in found_threshold_list if fpr > 0.0]

            # If there are positive numbers in the list, pick threshold for smallest pos number
            # Otherwise pick the threshold for the negative diff value closest to 0
            if len(pos_diffs := [(d, t) for d, t in diffs if d >= 0]) > 0:
                threshold = min(pos_diffs)[1]
            else:
                threshold = max(diffs)[1]

            break

    return threshold, compute_fpr(y_scores, threshold)


def compute_thresholds(df, fpr=0.05, epsilon=0.0005, per_domain_tuning=True):
    if not per_domain_tuning:
        return find_threshold(df, fpr, epsilon)

    thresholds = {}
    true_fprs = {}
    for d in df.domain.unique():
        t, true_fpr = find_threshold(df[df["domain"] == d], fpr, epsilon)
        thresholds[d] = t
        true_fprs[d] = true_fpr

    return thresholds, true_fprs


def get_unique_items(df, column, include_all=True):
    return df[column].unique().tolist() + ["all"] if include_all else df[column].unique().tolist()


def compute_scores(df, thresholds, require_complete=True, include_all=True):
    # Initialize the list of records for the scores
    scores = []

    # Filter out human data
    df = df[df["model"] != "human"]

    # For each domain, attack, model, and decoding strategy, filter the dataset
    for d in get_unique_items(df, "domain", include_all):
        dfd = df[df["domain"] == d] if d != "all" else df
        for a in get_unique_items(df, "attack", include_all):
            dfa = dfd[dfd["attack"] == a] if a != "all" else dfd
            for m in get_unique_items(df, "model", include_all):
                dfm = dfa[dfa["model"] == m] if m != "all" else dfa
                for s in get_unique_items(df, "decoding", include_all):
                    dfs = dfm[dfm["decoding"] == s] if s != "all" else dfm
                    for r in get_unique_items(df, "repetition_penalty", include_all):
                        df_filter = dfs[dfs["repetition_penalty"] == r] if r != "all" else dfs

                        # If no outputs for this split, continue
                        if len(df_filter) == 0:
                            continue

                        # If we're requiring all scores to be present and there are null scores, continue
                        if require_complete and (len(df_filter[df_filter["score"].isnull()]) > 0):
                            continue

                        # Remove null scores from the dataframe
                        df_filter = df_filter[df_filter["score"].notnull()]

                        # Initialize predictions
                        preds = []

                        # For each domain in df_filter
                        for domain in df_filter.domain.unique():
                            # Filter the dataset to just that domain
                            df_domain = df_filter[df_filter["domain"] == domain]

                            # Select the domain-specific threshold to use for classification
                            # (If thresholds is a dict, use the domain-specific threshold)
                            t = thresholds[domain] if type(thresholds) == dict else thresholds

                            # Get the 0 to 1 scores for the detector
                            y_model = df_domain["score"].to_numpy()

                            # Threshold scores using the threshold for this detector
                            # Source: https://stackoverflow.com/a/45648782
                            y_pred = (y_model >= t).astype(int)

                            # Add the prediction array to the list of predictions
                            preds.append(y_pred)

                        # Concatenate the predictions together
                        y_pred = np.concatenate(preds, axis=0)
                        y_true = np.ones(len(y_pred))

                        # Calculate the true positives and false negatives
                        tp = y_pred.sum()
                        fn = len(y_pred) - tp

                        # Compute accuracy and add to scores
                        scores.append(
                            {
                                "domain": d,
                                "model": m,
                                "decoding": s,
                                "repetition_penalty": r,
                                "attack": a,
                                "tp": int(tp),
                                "fn": int(fn),
                                "accuracy": accuracy_score(y_true, y_pred),
                            }
                        )
    return scores


def run_evaluation(
    results, df, target_fpr=0.05, epsilon=0.0005, per_domain_tuning=True, require_complete=True, include_all=True
):
    # Add detector outputs into a 'score' column
    df = load_detection_result(df, results)

    # Find thresholds per-domain for target FPR
    thresholds, fprs = compute_thresholds(df, target_fpr, epsilon, per_domain_tuning)

    # Compute accuracy scores for each split of the data
    scores = compute_scores(df, thresholds, require_complete)

    return {"scores": scores, "thresholds": thresholds, "fpr": fprs, "target_fpr": target_fpr}

def run_my_evaluation(results, df):
    # Add detector outputs into a 'score' column
    df = load_detection_result(df, results)

    best_threshold, best_accuracy, conf_matrix = find_my_threshold(df)
    total = sum(sum(conf_matrix))
    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")
    return {
            "threshold": best_threshold, 
            "accuracy": best_accuracy,
            "true_negative": {"count": int(conf_matrix[0][0]), "rate": float(conf_matrix[0][0] / total)},
            "false_positive": {"count": int(conf_matrix[0][1]), "rate": float(conf_matrix[0][1] / total)},
            "false_negative": {"count": int(conf_matrix[1][0]), "rate": float(conf_matrix[1][0] / total)},
            "true_positive": {"count": int(conf_matrix[1][1]), "rate": float(conf_matrix[1][1] / total)},
            "total": int(sum(sum(conf_matrix)))
            }


def find_my_threshold(df):
    y_pred = df["score"].tolist()
    y_pred = np.array(y_pred)

    #y_true will be 0 for all human data, but 1 for all machine data
    y_true = [0] * len(y_pred)
    y_true = np.array(y_true)
    y_true[df["model"] != "human"] = 1


    best_accuracy = 0
    best_threshold = 0
    #find min and max from y_pred
    min_val = min(y_pred)
    max_val = max(y_pred)
    thresholds = np.linspace(min_val, max_val, 100)

    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)
        accuracy = np.mean(y_pred_binary == y_true)  # Compute accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
            conf_matrix = confusion_matrix(y_true, y_pred_binary)
            #print(f"Threshold: {t}, Accuracy: {accuracy}, Confusion Matrix: {conf_matrix}")


    return best_threshold, best_accuracy, conf_matrix

def pred_count_for_hybrid(df, threshold):
    y_pred = df["score"].tolist()
    labels = []
    temp_dict = {}
    for y in y_pred:
        if y >= threshold:
            labels.append("AI")
        else:
            labels.append("human")
    
    #Count the number of human and AI labels
    for item in set(labels):
        #print(f"{item}: {labels.count(item)}")
        temp_dict[item] = {'count': labels.count(item), 'rate': labels.count(item) / len(labels)}
    return temp_dict


def run_evaluation_for_hybrid(results, df, threshold):
    # Add detector outputs into a 'score' column
    df = load_detection_result(df, results)

    # Compute accuracy scores for each split of the data
    pred_label_count = pred_count_for_hybrid(df, threshold)

    return {"prediction_labels": pred_label_count, "threshold": threshold}
