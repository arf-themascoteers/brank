from ds_manager import DSManager
from train_test_evaluator import evaluate_train_test_pair


def evaluate(dataset, folds, bands):
    oas = []
    ks = []
    d = DSManager(dataset,folds)
    for fold, splits in enumerate(d.get_k_folds()):
        evaluation_train_x = splits.evaluation_train_x[:,bands]
        evaluation_test_x = splits.evaluation_test_x[:,bands]

        oa, k = evaluate_train_test_pair(evaluation_train_x, splits.evaluation_train_y, evaluation_test_x, splits.evaluation_test_y)
        oas.append(oa)
        ks.append(k)
    return oas, ks


def compare(dataset, folds, bands1, bands2):
    oas1, ks1 = evaluate(dataset, folds, bands1)
    print("oas1,ks1",oas1,ks1)
    oas2, ks2 = evaluate(dataset, folds, bands2)
    print("oas2,ks2", oas2, ks2)

    mean_oas1 = sum(oas1)/len(oas1)
    mean_oas2 = sum(oas2)/len(oas2)

    if mean_oas1 > mean_oas2:
        print(f"First is better")
    else:
        print(f"Second is better")

    print(oas1, ks1)
    print(oas2, ks2)

    mean_k1 = sum(ks1)/len(ks1)
    mean_k2 = sum(ks2)/len(ks2)

    print(mean_oas1, mean_oas2)
    print(mean_k1, mean_k2)

dataset = "indian_pines"
folds = 10
bands1 = [10,16,24,27,40,47,53,59,63,78,83,92,100,106,119,120,129,140,145,153,154,166,176,182,190]
bands2 = [165,38,51,65,12,100,0,71,5,60,88,26,164,75,74,52,22,94,35,11,184,179,34,160,46]

compare(dataset, folds, bands1, bands2)



