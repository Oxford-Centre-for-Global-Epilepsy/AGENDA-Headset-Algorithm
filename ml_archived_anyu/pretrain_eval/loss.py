from ml_tflm.training.loss import StructureAwareLoss
from ml_tflm.training.train_utils import load_label_config
from ml_tflm.pre_training.loss_pretrain import LabelSupConLoss, InstanceSupConLoss

label_json_file = "ml_tflm/training/label_map.JSON"

def get_cls_loss(label_json_file):
    label_config = load_label_config(label_json_file)

    return StructureAwareLoss(label_config, clip_value=100.0, temperature=0.5)

def get_pre_loss(supervision, temperature, vicreg_weight):
    if supervision:
        return LabelSupConLoss(temperature=temperature, vicreg_weight=vicreg_weight)
    else:
        return InstanceSupConLoss(temperature=temperature, vicreg_weight=vicreg_weight)
    
if __name__ == "__main__":
    loss1 = get_cls_loss(label_json_file)

    loss2 = get_pre_loss(supervision=True, temperature=0.1, vicreg_weight=0.5)
    loss3 = get_pre_loss(supervision=False, temperature=0.1, vicreg_weight=0.5)

    print(0)