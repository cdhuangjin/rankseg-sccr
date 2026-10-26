import numpy as np
CONFIG={"ignore_index":255,"num_classes":21}

def confusion(prediction, label):
    valid = label != CONFIG["ignore_index"]
    assert np.all((prediction >= 0) & (prediction < CONFIG["num_classes"]))
    values = label[valid].astype(np.int64) * CONFIG["num_classes"] + prediction[valid]
    return np.bincount(values, minlength=CONFIG["num_classes"] ** 2).reshape(CONFIG["num_classes"], -1)

def scores(matrix):
    matrix = matrix.astype(np.float64)
    tp = matrix.diagonal()
    ground_truth = matrix.sum(1)
    prediction = matrix.sum(0)
    union = ground_truth + prediction - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    dice = np.divide(2 * tp, ground_truth + prediction, out=np.zeros_like(tp), where=(ground_truth + prediction) > 0)
    present = ground_truth > 0
    both = union > 0
    return {"miou_image_gt_present": float(100 * iou[present].mean()),
            "mdice_image_gt_present": float(100 * dice[present].mean()),
            "miou_image_union_present": float(100 * iou[both].mean()),
            "mdice_image_union_present": float(100 * dice[both].mean())}
