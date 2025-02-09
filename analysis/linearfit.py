import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
def get_pred(csv_file, esti_area = "esti_area", esti_individual_area = "esti_individual_area", esti_count = "esti_count"):
    include_list = [f.split('.')[0].replace('_jpg', '.jpg') for f in os.listdir("/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_dataset/revision_noaug/test/images")+os.listdir("/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_dataset/revision_noaug/test/images")]
    gt = pd.read_csv("gt_weights.csv")
    esti = pd.read_csv(csv_file)
    print(esti.keys())
    print(gt.keys())
    merged = pd.merge(gt,esti,on="file_name")
    print(csv_file, merged.keys(), merged.info())
    merged = merged[merged['file_name'].isin(include_list)]
    coeffs = np.polyfit(merged["avg_weight"], merged[esti_area], 1)
    print(coeffs)
    y_pred = np.polyval(coeffs, merged["avg_weight"])

    # Calculate R^2
    ss_res = np.sum((merged[esti_area] - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((merged[esti_area] - np.mean(merged[esti_area])) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)

    # count check
    miscount = merged["count"] - merged[esti_count]
    miscount = miscount.replace(0, np.nan)
    miscount = miscount.dropna(how='all', axis=0)
    miscount = miscount.replace(np.nan, 0)

    # weight error
    pred_weight = (merged[esti_area] - coeffs[1]) / coeffs[0]
    error = np.abs(merged["avg_weight"] - pred_weight)
    percent_error = 100*error/merged["avg_weight"]
    print(np.mean(error), np.mean(percent_error))
    return r2, y_pred, merged[esti_area], merged["avg_weight"], merged[esti_individual_area], miscount
def individual_matching(csv_file):
    gt = pd.read_csv("ml_human_level.csv")
    esti = pd.read_csv(csv_file)
    merged = pd.merge(gt,esti,on="file_name")
    include_list = [f.split('.')[0].replace('_jpg', '.jpg') for f in os.listdir("/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_dataset/revision_noaug/test/images")+os.listdir("/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_dataset/revision_noaug/test/images")]
    merged = merged[merged['file_name'].isin(include_list)]

    esti_boxes = merged["radii_centers"]
    gt_boxes = merged["gt_radii_centers"]
    esti_area = merged["esti_individual_area"]
    gt_area = merged["gt_individual_area"]
    import ast
    import cv2

    sizes = []
    for index, row in merged.iterrows():
        esti_boxes  = np.array(ast.literal_eval(row["radii_centers"])).reshape((-1,2))
        gt_boxes    = np.array(ast.literal_eval(row["gt_radii_centers"])).reshape((-1,2))
        esti_area   = np.array(ast.literal_eval(row["esti_individual_area"]))
        print(esti_area)
        gt_area     = np.array(ast.literal_eval(row["gt_individual_area"]))
        print(row["file_name"],esti_boxes.shape, gt_boxes.shape, row["esti_count"], row["gt_count"])
        empty_esti = np.zeros((640,640,3))
        empty_overlap = np.zeros((640,640,3))
        empty_gt = np.zeros((640,640,3))
        flag = False
        for est_ind, esti_center in enumerate(esti_boxes):
            min_dist = 10000
            min_ind = 0
            
            for gt_ind, gt_center in enumerate(gt_boxes):
                dist = np.sqrt(np.sum((esti_center-gt_center)**2))
                cv2.circle(empty_gt, (gt_center*640).astype(np.uint32), int(gt_area[gt_ind]), color=(0,0,255), thickness=5)
                if dist < min_dist:
                    min_dist = dist
                    min_ind = gt_ind
            if min_dist > 0.01:
                flag = False
            else:
                sizes.append([gt_area[min_ind], esti_area[est_ind]])
            #print(min_ind, min_dist, (esti_center*640).astype(np.uint32))
            cv2.circle(empty_esti, (esti_center*640).astype(np.uint32), int(esti_area[est_ind]), color=(255,0,0), thickness=5)
            cv2.circle(empty_overlap, (gt_boxes[min_ind]*640).astype(np.uint32), int(gt_area[min_ind]), color=(0,255,0), thickness=5)
        if flag:
            plt.subplot(131)
            plt.imshow(empty_esti)
            plt.subplot(132)
            plt.imshow(empty_gt)
            plt.subplot(133)
            plt.imshow(empty_overlap)
            plt.show()
    return np.array(sizes)
result_v5 = get_pred("ml_v5m_50.csv")
# result_v5n = get_pred("ml_v5n_50.csv")
# result_v5ghbi = get_pred("ml_v5ghbi_50.csv")
# result_v8 = get_pred("ml_v8m_50.csv")
# result_v11 = get_pred("ml_v11m_50.csv")
result_human = get_pred("ml_human_level.csv", esti_individual_area="gt_individual_area", esti_count="gt_count")
result_trad1 = get_pred("ml_traditional.csv", esti_area="HT_based_nofilter", esti_individual_area="miscounts")
result_trad2 = get_pred("ml_traditional.csv", esti_area="HT-based", esti_individual_area="miscounts")
#print(f"v5m R2:{result_v5[0]}\nv8m R2: {result_v8[0]}\nv11m R2: {result_v11[0]}\nhuman R2: {result_human[0]}\HT_based_nofilter R2: {result_trad1[0]}\HT_based R2: {result_trad2[0]}")
print(result_v5[-1])
print(result_trad1[-1])
print(result_trad2[-1])
# print(result_v8[-1])
# print(result_v11[-1])
# print(result_human[-1])
plt.figure(figsize=(6,4))
m_size = 12

# plt.scatter(result_v8[3], result_v8[2], label=f"YOLOv8: $R^2$={result_v8[0]:.3f}",marker='^',s=m_size,c="violet")
# plt.plot(result_v8[3], result_v8[1],alpha=0.3)
# plt.scatter(result_v11[3], result_v11[2], label=f"YOLOv11: $R^2$={result_v11[0]:.3f}",marker='+',s=m_size,c="magenta")
# plt.plot(result_v11[3], result_v11[1],alpha=0.3)
plt.scatter(result_v5[3], result_v5[2], label=f"YOLOv5: $R^2$={result_v5[0]:.3f}",marker='*',s=m_size,c="red")
plt.plot(result_v5[3], result_v5[1],alpha=0.3)
plt.scatter(result_human[3], result_human[2], label=f"human: $R^2$={result_human[0]:.3f}",marker='o',s=m_size,c="black")
plt.plot(result_human[3], result_human[1],alpha=0.3)
plt.scatter(result_trad1[3], result_trad1[2], label=f"HT_based_nofilter: $R^2$={result_trad1[0]:.3f}",marker='<',s=m_size,c="green")
plt.plot(result_trad1[3], result_trad1[1],alpha=0.3)
plt.scatter(result_trad2[3], result_trad2[2], label=f"HT_based: $R^2$={result_trad2[0]:.3f}",marker='>',s=m_size,c="blue")
plt.plot(result_trad2[3], result_trad2[1],alpha=0.3)

plt.xlabel("Measured average berry weight (g)")
plt.ylabel("Estimated average berry size (cm$^2$)")
plt.grid(alpha=0.2)
plt.ylim([0,5])
plt.legend(
    title="",
    title_fontsize=10,
    bbox_to_anchor=(0.5, 1.05),
    loc='center',
    ncol=2,
    frameon=False
)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig("fig1.png",dpi=600, bbox_inches='tight')
plt.show()

import ast
# plt.subplot(221)
# parsed_arrays = [ast.literal_eval(item) for item in np.array(result_v5[-2])]
# flattened_array = np.concatenate([np.array(sub_array) for sub_array in parsed_arrays])
# plt.hist(flattened_array,bins=50,alpha=0.6,label="v5")
# plt.legend()
# plt.subplot(222)
# parsed_arrays = [ast.literal_eval(item) for item in np.array(result_v8[-2])]
# flattened_array = np.concatenate([np.array(sub_array) for sub_array in parsed_arrays])
# plt.hist(flattened_array,bins=50,alpha=0.6,label="v8")
# plt.legend()
# plt.subplot(223)
# parsed_arrays = [ast.literal_eval(item) for item in np.array(result_v11[-2])]
# flattened_array = np.concatenate([np.array(sub_array) for sub_array in parsed_arrays])
# plt.hist(flattened_array,bins=50,alpha=0.6,label="v11")
# plt.subplot(224)
# parsed_arrays = [ast.literal_eval(item) for item in np.array(result_human[-2])]
# flattened_array = np.concatenate([np.array(sub_array) for sub_array in parsed_arrays])
# plt.hist(flattened_array,bins=50,alpha=0.6,label="human")
# plt.legend()
# plt.show()

parsed_arrays = [ast.literal_eval(item) for item in np.array(result_human[-2])]
flattened_array = np.concatenate([np.array(sub_array) for sub_array in parsed_arrays])
plt.figure(figsize=(2.5,3.5))
plt.hist(flattened_array,bins=50,alpha=0.6,label="human")
plt.ylabel("Counts")
plt.xlabel("Individual Blueberry Size (cm$^2$)")
plt.savefig("fig2.png",dpi=600, bbox_inches='tight')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

sizes = individual_matching("ml_v5m_50.csv")
sizes_df = pd.DataFrame(sizes)
print(sizes_df.describe())
coeffs = np.polyfit(sizes[:,0], sizes[:,1], 1)
print(coeffs)
y_pred = np.polyval(coeffs, sizes[:,0])
ss_res = np.sum((sizes[:,1] - y_pred) ** 2)  # Residual sum of squares
ss_tot = np.sum((sizes[:,1] - np.mean(sizes[:,1])) ** 2)  # Total sum of squares
r2 = 1 - (ss_res / ss_tot)
print(sizes.shape)
#using_hist2d(fig, sizes[:,0], sizes[:,1])
plt.figure(figsize=(6,4))
plt.plot(np.linspace(0,6,10),np.linspace(0,6,10),c="red", alpha = 0.5, label = "y=x")
plt.plot(sizes[:,0], y_pred, color = 'green', alpha = 0.5, label = f"R$^2$={r2:0.2f}")
plt.scatter(sizes[:,0], sizes[:,1],alpha=0.2)#, bins=(100, 50), cmap=plt.cm.jet, cmin=1)
#plt.hist2d(sizes[:,0], sizes[:,1], bins=(100, 50), cmap=plt.cm.jet, cmin=1)
plt.legend()
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('Blueberry counts')
plt.xlabel("Labeled blueberry size (cm$^2$)")
plt.ylabel("Estimated blueberry size (cm$^2$)")
plt.xlim([0,6])
plt.ylim([0,6])

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(alpha=0.2)
plt.savefig("fig3.png",dpi=600, bbox_inches='tight')


plt.show()