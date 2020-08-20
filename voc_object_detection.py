import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_object_detection as tf_obj_detector

from PIL import Image
import matplotlib.pyplot as plt

# Custom function to parse the data. #
def _parse_image(
    filename, img_rows=448, img_cols=448):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_rows, img_cols, 3))
    return tf.expand_dims(image_resized, axis=0)

def train(
    voc_model, sub_batch_sz, batch_size, 
    batch_test, train_dataset, valid_dataset, 
    img_disp, img_box, start_step, max_steps, 
    optimizer, ckpt, ck_manager, init_lr=1.0e-3, 
    decay=0.75, display_step=100, step_cool=50, 
    save_flag=False):
    n_data  = len(train_dataset)
    n_batch = 0
    upd_epoch = int(n_data/batch_size)
    
    start_time = time.time()
    tot_reg_loss  = 0.0
    tot_cls_loss  = 0.0
    for step in range(start_step, max_steps):
        batch_sample = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        img_files = [train_dataset[x][0] for x in batch_sample]
        img_batch = tf.concat(
            [_parse_image(x) for x in img_files], axis=0)
        
        img_bbox = tf.concat([
            tf.expand_dims(tf.sparse.to_dense(
                train_dataset[x][1], validate_indices=False), 
                    axis=0) for x in batch_sample], axis=0)
        img_bbox = tf.cast(img_bbox, tf.float32)
        img_mask = img_bbox[:, :, :, :, 4]
        
        epoch = int(step * batch_size / n_data)
        lrate = max(decay**epoch * init_lr, 1.0e-4)
        
        tmp_losses = tf_obj_detector.train_step(
            voc_model, sub_batch_sz, 
            img_batch, img_bbox, img_mask, 
            optimizer, learning_rate=lrate)
        
        ckpt.step.assign_add(1)
        n_batch += 1
        tot_cls_loss += tmp_losses[0]
        tot_reg_loss += tmp_losses[1]
        if n_batch % display_step == 0:
            print("Intermediate Loss at Epoch", 
                  str(epoch+1), "step", str(step+1)+":", 
                  str(tot_cls_loss.numpy()/n_batch)+",", 
                  str(tot_reg_loss.numpy()/n_batch))
        
        if (step+1) % upd_epoch == 0:
            avg_reg_loss = tot_reg_loss.numpy() / upd_epoch
            avg_cls_loss = tot_cls_loss.numpy() / upd_epoch
            tot_reg_loss = 0.0
            tot_cls_loss = 0.0
            
            # Validation. #
            n_batch = 0
            valid_error = 0.0
            valid_foc   = 0.0
        
            n_img = len(valid_dataset)
            if n_img <= batch_test:
                n_test_batch = 1
            elif n_img % batch_test == 0:
                n_test_batch = n_img / batch_test
            else:
                n_test_batch = n_img // batch_test + 1
            
            n_test_batch = int(n_test_batch)
            for n_batch in range(n_test_batch):
                id_st = n_batch * batch_test
                if n_batch == (n_test_batch-1):
                    id_end = n_img
                else:
                    id_end = (n_batch+1) * batch_test
                
                img_batch = tf.concat([_parse_image(
                    valid_dataset[x][0]) for x in \
                        range(id_st, id_end)], axis=0)
                
                img_bbox = tf.concat([
                    tf.expand_dims(tf.sparse.to_dense(
                        valid_dataset[x][1], validate_indices=False), 
                            axis=0) for x in range(id_st, id_end)], axis=0)
                img_mask = img_bbox[:, :, :, :, 4]
                
                tmp_output = voc_model.predict(img_batch)
                reg_output = tmp_output[:, :, :, :, :4]
                cls_output = tmp_output[:, :, :, :, 4:]
                
                correct_reg = img_bbox[:, :, :, :, :4]
                correct_cls = img_bbox[:, :, :, :, 4:]
                
                valid_foc += np.sum(
                    tf_obj_detector.one_class_focal_loss(
                        correct_cls, cls_output))
                
                valid_error += np.sum(np.multiply(
                    np.expand_dims(img_mask, axis=4), 
                    np.abs(reg_output - correct_reg)))
            
            print("Epoch", str(epoch+1), "Summary:")
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Average Epoch Reg. Loss:", str(avg_reg_loss) + ".")
            print("Average Epoch Focal Loss:", str(avg_cls_loss) + ".")
            print("Validation Abs. Error:", str(valid_error / n_img))
            print("Validation Focal Loss:", str(valid_foc / n_img))
            
            elapsed_time = (time.time() - start_time) / 60.0
            print("Elapsed time:", str(elapsed_time), "mins.")
            
            if save_flag:
                # Save the model. #
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
                print("-" * 50)
            
            tf_obj_detector.obj_detect_results(
                img_disp, voc_model, label_dict, 
                heatmap=True, img_box=img_box, thresh=0.50)
            start_time = time.time()
        
        if (step+1) % step_cool == 0:
            time.sleep(180)

# Load the VOC 2012 dataset. #
tmp_pd_file = "C:/Users/admin/Desktop/" +\
    "Data/VOCdevkit/VOC2012/voc_2012_objects.csv"
raw_voc_df  = pd.read_csv(tmp_pd_file)

tmp_df_cols = ["filename", "width", "height", 
               "xmin", "xmax", "ymin", "ymax", "label"]
raw_voc_df  = pd.DataFrame(raw_voc_df, columns=tmp_df_cols)
image_files = sorted(list(pd.unique(raw_voc_df["filename"])))
image_files = pd.DataFrame(image_files, columns=["filename"])
print("Total of", str(len(image_files)), "images in VOC dataset.")

# The VOC 2012 class names. #
class_names = list(
    sorted(list(pd.unique(raw_voc_df["label"]))))
class_names = ["background"] + [str(x) for x in class_names]
class_dict  = dict(
    [(class_names[x], x) for x in range(len(class_names))])
label_dict  = dict(
    [(x, class_names[x]) for x in range(len(class_names))])

raw_voc_df["img_label"] = [
    class_dict.get(str(
        raw_voc_df.iloc[x]["label"])) \
            for x in range(len(raw_voc_df))]
raw_voc_df["x_centroid"] = \
    (raw_voc_df["xmax"] + raw_voc_df["xmin"]) / 2
raw_voc_df["y_centroid"] = \
    (raw_voc_df["ymax"] + raw_voc_df["ymin"]) / 2

# Split into train and validation data. #
prop_tr = 0.9
n_train = int(prop_tr * len(image_files))

np.random.seed(1234)
id_rand = np.random.permutation(n_train)
id_perm = np.random.permutation(len(image_files))

train_img = image_files.iloc[id_perm[:n_train]]
valid_img = image_files.iloc[id_perm[n_train:]]

# Define the Neural Network. #
restore_flag = True
subsample  = False
img_width  = 448
img_height = 448
n_filters  = 16
step_cool  = 250

init_lr    = 0.001
decay_rate = 1.0
max_steps  = 25000
batch_size = 48
sub_batch  = 12
batch_test = 10
n_classes  = len(class_names)

display_step = 50
down_width   = int(img_width/8)
down_height  = int(img_height/8)
dense_shape  = [down_height, down_width, 4, n_classes+4]

# Define the checkpoint callback function. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
train_loss  = voc_path + "voc_losses_v4.csv"
ckpt_path   = voc_path + "voc_v4.ckpt"
ckpt_dir    = os.path.dirname(ckpt_path)
ckpt_model  = voc_path + "voc_keras_model_v4"

# Load the weights if continuing from a previous checkpoint. #
voc_model = \
    tf_obj_detector.build_model(n_filters, n_classes)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    voc_model=voc_model, 
    optimizer=optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

if restore_flag:
    checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        print("Model restored from {}".format(
            ck_manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
st_step = checkpoint.step.numpy().astype(np.int32)

# Print out the model summary. #
print(voc_model.summary())
print("-" * 50)

# Find a way to remove duplicate indices from the data. #
# Total output classes is n_classes + centerness (1) + regression (4). #
print("Formatting the object detection bounding boxes.")
start_time = time.time()

train_list = []
for n_img in range(len(train_img)):
    tot_obj  = 0
    img_file = train_img.iloc[n_img]["filename"]
    
    tmp_filter = raw_voc_df[
        raw_voc_df["filename"] == img_file]
    tmp_filter = tmp_filter[[
        "width", "height", "img_label", "xmin", "xmax", 
        "ymin", "ymax", "x_centroid", "y_centroid"]]
    
    tmp_w_ratio = tmp_filter.iloc[0]["width"] / img_width
    tmp_h_ratio = tmp_filter.iloc[0]["height"] / img_height
    if len(tmp_filter) > 0:
        tmp_indices = []
        tmp_values  = []
        for n_obj in range(len(tmp_filter)):
            tmp_object = tmp_filter.iloc[n_obj]
            tmp_label  = int(tmp_object["img_label"])
            
            tmp_w_min = tmp_object["xmin"] / tmp_w_ratio
            tmp_w_max = tmp_object["xmax"] / tmp_w_ratio
            tmp_h_min = tmp_object["ymin"] / tmp_h_ratio
            tmp_h_max = tmp_object["ymax"] / tmp_h_ratio
            
            tmp_width  = tmp_w_max - tmp_w_min
            tmp_height = tmp_h_max - tmp_h_min
            
            if tmp_width < 0 or tmp_height < 0:
                continue
            elif tmp_width < 64 and tmp_height < 64:
                id_sc = 0
                box_scale = 64
            elif tmp_width < 128 and tmp_height < 128:
                id_sc = 1
                box_scale = 128
            elif tmp_width < 256 and tmp_height < 256:
                id_sc = 2
                box_scale = 256
            else:
                id_sc = 3
                box_scale = max(img_width, img_height)
            
            # The regression and classification outputs #
            # are 104 x 104 pixels, so divide by 8.     #
            tmp_x_cen = tmp_object["x_centroid"] / tmp_w_ratio
            tmp_y_cen = tmp_object["y_centroid"] / tmp_h_ratio
            tmp_w_cen = int(tmp_x_cen/8)
            tmp_h_cen = int(tmp_y_cen/8)
            
            tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
            tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
            tmp_w_reg = (tmp_w_max - tmp_w_min) / box_scale
            tmp_h_reg = (tmp_h_max - tmp_h_min) / box_scale
            
            val_array = [tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1]
            for dim_arr in range(5):
                tmp_values.append(val_array[dim_arr])
                tmp_indices.append([tmp_h_cen, tmp_w_cen, id_sc, dim_arr])
            
            id_obj = tmp_label + 4
            tmp_values.append(1)
            tmp_indices.append([tmp_h_cen, tmp_w_cen, id_sc, id_obj])
        
        tmp_sparse_tensor = tf.sparse.SparseTensor(
            tmp_indices, tmp_values, dense_shape)
        train_list.append((img_file, tmp_sparse_tensor))

# Validation dataset. #
valid_list = []
for n_img in range(len(valid_img)):
    tot_obj  = 0
    img_file = train_img.iloc[n_img]["filename"]
    
    tmp_filter = raw_voc_df[
        raw_voc_df["filename"] == img_file]
    tmp_filter = tmp_filter[[
        "width", "height", "img_label", "xmin", "xmax", 
        "ymin", "ymax", "x_centroid", "y_centroid"]]
    
    tmp_w_ratio = tmp_filter.iloc[0]["width"] / img_width
    tmp_h_ratio = tmp_filter.iloc[0]["height"] / img_height
    if len(tmp_filter) > 0:
        tmp_indices = []
        tmp_values  = []
        for n_obj in range(len(tmp_filter)):
            tmp_object = tmp_filter.iloc[n_obj]
            tmp_label  = int(tmp_object["img_label"])
            
            tmp_w_min = tmp_object["xmin"] / tmp_w_ratio
            tmp_w_max = tmp_object["xmax"] / tmp_w_ratio
            tmp_h_min = tmp_object["ymin"] / tmp_h_ratio
            tmp_h_max = tmp_object["ymax"] / tmp_h_ratio
            
            tmp_width  = tmp_w_max - tmp_w_min
            tmp_height = tmp_h_max - tmp_h_min
            
            if tmp_width < 0 or tmp_height < 0:
                continue
            elif tmp_width < 64 and tmp_height < 64:
                id_sc = 0
                box_scale = 64
            elif tmp_width < 128 and tmp_height < 128:
                id_sc = 1
                box_scale = 128
            elif tmp_width < 256 and tmp_height < 256:
                id_sc = 2
                box_scale = 256
            else:
                id_sc = 3
                box_scale = max(img_width, img_height)
            
            # The regression and classification outputs #
            # are 104 x 104 pixels, so divide by 8.     #
            tmp_x_cen = tmp_object["x_centroid"] / tmp_w_ratio
            tmp_y_cen = tmp_object["y_centroid"] / tmp_h_ratio
            tmp_w_cen = int(tmp_x_cen/8)
            tmp_h_cen = int(tmp_y_cen/8)
            
            tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
            tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
            tmp_w_reg = (tmp_w_max - tmp_w_min) / box_scale
            tmp_h_reg = (tmp_h_max - tmp_h_min) / box_scale
            
            val_array = [tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1]
            for dim_arr in range(5):
                tmp_values.append(val_array[dim_arr])
                tmp_indices.append([tmp_h_cen, tmp_w_cen, id_sc, dim_arr])
            
            id_obj = tmp_label + 4
            tmp_values.append(1)
            tmp_indices.append([tmp_h_cen, tmp_w_cen, id_sc, id_obj])
        
        tmp_sparse_tensor = tf.sparse.SparseTensor(
            tmp_indices, tmp_values, dense_shape)
        valid_list.append((img_file, tmp_sparse_tensor))

elapsed_tm = (time.time() - start_time) / 60
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

if subsample:
    train_list = train_list[:1000]
    valid_list = valid_list[:100]

print("Fit model on training data (" +\
      str(len(train_list)) + " training samples).")
img_disp = train_list[2][0]
img_box  = tf.sparse.to_dense(train_list[2][1])

fig, ax = plt.subplots(1)
tmp_img = np.array(
    Image.open(img_disp), dtype=np.uint8)
ax.imshow(tmp_img)
fig.savefig("test_image.jpg", dpi=199)

plt.close()
del fig, ax

train(voc_model, sub_batch, batch_size, batch_test, 
      train_list, valid_list, img_disp, img_box, 
      st_step, max_steps, optimizer, checkpoint, 
      ck_manager, init_lr=init_lr, decay=decay_rate, 
      step_cool=step_cool, display_step=display_step, 
      save_flag=True)
print("Model fitted.")

save_path = ck_manager.save()
print("Saved model to {}".format(save_path))
