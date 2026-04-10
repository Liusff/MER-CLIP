import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
'''
def plot_confusion_matrix(gt, pred, label_map, name):
    cm = confusion_matrix(gt, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

    labels = [label_map[str(i)] for i in range(len(label_map))]

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm_normalized, cmap='Blues')  # Display matrix
    fig.colorbar(cax)

    # Set labels for axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Loop over data dimensions and create text annotations
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='black' if cm_normalized[i, j] < 0.5 else 'white')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #plt.title('Confusion Matrix')
    plt.tight_layout()  # Ensure everything fits without overlap
    plt.savefig(f"tools/confusion_matrix/{name}.jpg")
    plt.show()
'''
def plot_confusion_matrix(gt, pred, label_map, name):
    plt.rcParams.update({'font.size': 24})  # 设置全局字体大小
    cm = confusion_matrix(gt, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    labels = [label_map[str(i)] for i in range(len(label_map))]

    # 增大画布尺寸
    fig, ax = plt.subplots(figsize=(12, 10))  # 调整figsize
    
    # 设置色条标签大小
    cax = ax.matshow(cm_normalized, cmap='Blues')
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=22)  # 色条文字大小

    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    
    # 调整刻度标签字体大小
    ax.set_xticklabels(labels, 
                      rotation=30, 
                      #ha='right',
                      fontsize=26)  # x轴标签大小
                      
    ax.set_yticklabels(labels,
                      fontsize=26)  # y轴标签大小

    # 调整注释文字大小
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha='center', 
                    va='center',
                    fontsize=38,  # 注释文字大小
                    color='black' if cm_normalized[i, j] < 0.5 else 'white')

    # 调整坐标轴标题大小
    #ax.xaxis.label.set_size(22)  # x轴标题
    #ax.yaxis.label.set_size(22)  # y轴标题
    
    plt.xlabel('Predicted Label', labelpad=15)  # 增加标签间距
    plt.ylabel('True Label', labelpad=15)
    
    plt.tight_layout()
    
    # 保存时提高分辨率
    plt.savefig(f"tools/confusion_matrix/{name}.jpg", 
               dpi=300,  # 提高输出分辨率
               bbox_inches='tight')
    plt.show()
def confusionMatrix_def(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred, label_map):
    unique_elements_gt = np.unique(final_gt)
    unique_elements_pred = np.unique(final_pred)
    # 如果数组中只有一个唯一元素，则表示只包含一类数字
    if len(unique_elements_gt) == 1 and len(unique_elements_pred) == 1 and unique_elements_gt[0] == unique_elements_pred[0]:
        UF1 = 1
        UAR = 1
        return UF1, UAR 

    label_dict = {v: int(k) for k, v in label_map.items()}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix_def(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return ' ',' '
path = "work_dirs/uf2_progres_clip+transformer_aug_clshead_samm_3cls2/results_8321_8434.csv"
results = pd.read_csv(path, header=None)
pred = results.iloc[:, 1]
gt = results.iloc[:, 2]
#label_map = {"0": "anger", "1": "contempt", "2": "disgust", "3": "fear", "4": "happiness", "5": "sadness", "6": "surprise"} #DFME
#label_map = {"0": "anger", "1": "disgust", "2": "fear", "3": "happy", "4": "others", "5": "sad", "6": "surprise"} #casme3
#label_map = {"0": "anger", "1": "contempt", "2": "happiness", "3": "other", "4": "surprise"}  #samm
#label_map = {"0": "negative", "1": "positive", "2": "surprise", "3": "others"} #casme3_4cls
label_map = {"0": "negative", "1": "positive", "2": "surprise"} #3_cls
#label_map = {"0": "disgust", "1": "happiness", "2": "others", "3": "repression", "4": "surprise"} #casme2
#label_map = {"0": "disgust", "1": "fear", "2": "happiness", "3": "others", "4": "surprise"} #mmew
#label_map = {"0": "disgust", "1": "happiness", "2": "others", "3": "surprise"}  #mmew_4cls
uf1, uar = recognition_evaluation(gt, pred, label_map)
acc = accuracy_score(gt, pred, normalize=True, sample_weight=None)
print("UF1 = ", uf1)
print("UAR = ", uar)
print("ACC = ", acc)
# Plot confusion matrix

name = path.split("/")[-1].split(".")[0]
print("Plot confusion matrix to ", name)
plot_confusion_matrix(gt, pred, label_map, name)


'''
def plot_confusion_matrix(gt, pred, label_map, name):
    cm = confusion_matrix(gt, pred)
    row_sums = cm.sum(axis=1, keepdims=True)  # Sum of each row

    # Avoid division by zero: set rows with sum zero to 1 temporarily
    row_sums[row_sums == 0] = 1
    cm_normalized = cm / row_sums  # Normalize by row
    print(cm_normalized)

    labels = [label_map[str(i)] for i in range(len(label_map))]
    annot = np.vectorize(lambda x: f"{x:.4f}")(cm_normalized)
    print(annot)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 10})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("tools/confusion_matrix/{}.jpg".format(name))
    plt.show()
'''
