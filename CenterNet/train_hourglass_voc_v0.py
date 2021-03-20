import time
import numpy as np
import pandas as pd
import pickle as pkl
from utils import convert_to_xywh

import tensorflow as tf
import tf_hourglass_net as tf_obj_detector

# Custom function to parse the data. #
def _parse_image(
    filename, img_rows, img_cols):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_rows, img_cols, 3))
    return image_resized

def image_augment(img_in, img_bbox, p=0.5):
    if np.random.uniform() >= p:
        p_tmp = np.random.uniform()
        if p_tmp <= 0.50:
            p_uni = np.random.uniform()
            if p_uni <= 0.50:
                tmp_in = tf.image.random_brightness(img_in, 0.25)
            else:
                tmp_in = tf.image.random_contrast(img_in, 0.75, 1.25)
            
            # Rescale the image values if it exceeds 1.0. #
            tmp_max = tf.reduce_max(tmp_in).numpy()
            if tmp_max > 1.0:
                tmp_in = tmp_in / tmp_max
            return (tf.constant(tmp_in), img_bbox)
        else:
            # Flip left-right. #
            img_in = img_in[:, ::-1, :]
            
            tmp_bbox = img_bbox
            img_bbox = tmp_bbox[:, ::-1, :, :]
            
            img_bbox[:, :, :, 1] = 1.0 - img_bbox[:, :, :, 1]
            return (img_in, tf.constant(img_bbox))
    else:
        return (img_in, img_bbox)

def train(
    voc_model, n_classes, sub_batch_sz, batch_size, 
    train_data, training_loss, st_step, max_steps, 
    optimizer, ckpt, ck_manager, label_dict, init_lr=1.0e-3, 
    min_lr=1.0e-6, decay=0.75, display_step=100, step_cool=50, 
    base_dims=None, disp_rows=320, disp_cols=320, thresh=0.50, 
    save_flag=False, train_loss_log="log_training_losses.csv"):
    n_data = len(train_data)
    min_scale  = min(disp_rows, disp_cols)
    disp_scale = [min_scale / (2**x) for x in range(4)]
    disp_scale = disp_scale[::-1]
    
    if base_dims is None:
        base_dims = [256, 320, 384, 448]
    
    start_time = time.time()
    tot_reg_loss = 0.0
    tot_cls_loss = 0.0
    for step in range(st_step, max_steps):
        batch_sample  = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        img_dims = np.random.choice(base_dims)
        img_boxes = []
        img_scale = [(img_dims / (2**x)) for x in range(4)]
        img_scale = img_scale[::-1]
        for tmp_idx in batch_sample:
            img_bbox  = np.zeros([
                int(img_dims/8), 
                int(img_dims/8), 4, n_classes+5])
            tmp_bbox  = np.array(train_data[
                tmp_idx]["objects"]["bbox"])
            tmp_bbox  = convert_to_xywh(tmp_bbox).numpy()
            tmp_class = np.array(train_data[
                tmp_idx]["objects"]["label"])
            
            # Sort by area in descending order. #
            tmp_box_areas = \
                tmp_bbox[:, 2] * tmp_bbox[:, 3] * 100
            
            obj_class  = \
                tmp_class[np.argsort(tmp_box_areas)]
            tmp_sorted = \
                tmp_bbox[np.argsort(tmp_box_areas)]
            for n_box in range(len(tmp_sorted)):
                tmp_object = tmp_sorted[n_box]
                
                tmp_x_cen  = tmp_object[0] * img_dims
                tmp_y_cen  = tmp_object[1] * img_dims
                tmp_width  = tmp_object[2] * img_dims
                tmp_height = tmp_object[3] * img_dims
                
                if tmp_width < 0 or tmp_height < 0:
                    continue
                elif tmp_width < img_scale[0] \
                    and tmp_height < img_scale[0]:
                    id_sc = 0
                    box_scale = img_scale[0]
                elif tmp_width < img_scale[1] \
                    and tmp_height < img_scale[1]:
                    id_sc = 1
                    box_scale = img_scale[1]
                elif tmp_width < img_scale[2] \
                    and tmp_height < img_scale[2]:
                    id_sc = 2
                    box_scale = img_scale[2]
                else:
                    id_sc = 3
                    box_scale = img_scale[3]
                
                tmp_w_reg = tmp_width / box_scale
                tmp_h_reg = tmp_height / box_scale
                tmp_w_cen = int(tmp_x_cen / 8)
                tmp_h_cen = int(tmp_y_cen / 8)
                tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
                tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
                cls_label = obj_class[n_box] + 5
                
                img_bbox[tmp_h_cen, tmp_w_cen, id_sc, :5] = [
                    tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1.0]
                img_bbox[tmp_h_cen, tmp_w_cen, id_sc, cls_label] = 1.0
            
            img_boxes.append(img_bbox)
            del img_bbox
            
            if tmp_idx == batch_sample[-1]:
                disp_box = np.zeros([
                    int(disp_rows/8), 
                    int(disp_cols/8), 4, n_classes+5])
                
                for n_box in range(len(tmp_sorted)):
                    tmp_object = tmp_sorted[n_box]
                    
                    tmp_x_cen  = tmp_object[0] * disp_rows
                    tmp_y_cen  = tmp_object[1] * disp_cols
                    tmp_width  = tmp_object[2] * disp_rows
                    tmp_height = tmp_object[3] * disp_cols
                    
                    if tmp_width < 0 or tmp_height < 0:
                        continue
                    elif tmp_width < disp_scale[0] \
                        and tmp_height < disp_scale[0]:
                        id_sc = 0
                        box_scale = disp_scale[0]
                    elif tmp_width < disp_scale[1] \
                        and tmp_height < disp_scale[1]:
                        id_sc = 1
                        box_scale = disp_scale[1]
                    elif tmp_width < disp_scale[2] \
                        and tmp_height < disp_scale[2]:
                        id_sc = 2
                        box_scale = disp_scale[2]
                    else:
                        id_sc = 3
                        box_scale = disp_scale[3]
                    
                    tmp_w_reg = tmp_width / box_scale
                    tmp_h_reg = tmp_height / box_scale
                    tmp_w_cen = int(tmp_x_cen / 8)
                    tmp_h_cen = int(tmp_y_cen / 8)
                    tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
                    tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
                    cls_label = obj_class[n_box] + 5
                    
                    disp_box[tmp_h_cen, tmp_w_cen, id_sc, :5] = [
                        tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1.0]
                    disp_box[tmp_h_cen, tmp_w_cen, id_sc, cls_label] = 1.0
        
        img_files = [
            train_data[x]["image"] for x in batch_sample]
        img_batch = [_parse_image(
            x, img_rows=img_dims, 
            img_cols=img_dims) for x in img_files]
        
        img_tuple = [image_augment(
            img_batch[x], img_boxes[x]) \
                for x in range(batch_size)]
        img_batch = [tf.expand_dims(
            x, axis=0) for x, y in img_tuple]
        img_bbox  = [tf.expand_dims(
            y, axis=0) for x, y in img_tuple]
        
        # Note that TF parses the image transposed, so the  #
        # bounding boxes coordinates are already transposed #
        # during the formatting of the data.                #
        img_batch = tf.concat(img_batch, axis=0)
        img_bbox  = tf.cast(tf.concat(
            img_bbox, axis=0), tf.float32)
        img_mask  = img_bbox[:, :, :, :, 4]
        
        epoch = int(step * batch_size / n_data)
        lrate = max(decay**epoch * init_lr, min_lr)
        
        tmp_losses = tf_obj_detector.train_step(
            voc_model, sub_batch_sz, img_batch, 
            img_bbox, img_mask, optimizer, learning_rate=lrate)
        
        ckpt.step.assign_add(1)
        tot_cls_loss += tmp_losses[0]
        tot_reg_loss += tmp_losses[1]
        
        if (step+1) % display_step == 0:
            avg_reg_loss = tot_reg_loss.numpy() / display_step
            avg_cls_loss = tot_cls_loss.numpy() / display_step
            training_loss.append((step+1, avg_cls_loss, avg_reg_loss))
            
            tot_reg_loss = 0.0
            tot_cls_loss = 0.0
            
            print("Step", str(step+1), "Summary:")
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Average Epoch Cls. Loss:", str(avg_cls_loss) + ".")
            print("Average Epoch Reg. Loss:", str(avg_reg_loss) + ".")
            
            elapsed_time = (time.time() - start_time) / 60.0
            print("Elapsed time:", str(elapsed_time), "mins.")
            
            start_time = time.time()
            if (step+1) % step_cool != 0:
                img_title = "CenterNet Object Detection Result "
                img_title += "at Step " + str(step+1)
                
                tf_obj_detector.show_object_boxes(
                    img_batch[-1], img_bbox[-1], img_dims)
                
                disp_box = tf.constant(disp_box)
                tf_obj_detector.obj_detect_results(
                    img_files[-1], voc_model, 
                    label_dict, heatmap=True, 
                    img_box=disp_box, thresh=thresh, 
                    img_rows=disp_rows, img_cols=disp_cols, 
                    img_scale=disp_scale, img_title=img_title)
                print("-" * 50)
        
        if (step+1) % step_cool == 0:
            if save_flag:
                # Save the training losses. #
                train_cols_df = ["step", "cls_loss", "reg_loss"]
                train_loss_df = pd.DataFrame(
                    training_loss, columns=train_cols_df)
                train_loss_df.to_csv(train_loss_log, index=False)
                
                # Save the model. #
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            print("-" * 50)
            
            save_img_file = "C:/Users/admin/Desktop/Data/"
            save_img_file += "Results/VOC_Object_Detection/"
            save_img_file += "voc_obj_detect_" + str(step+1) + ".jpg"
            
            img_title = "CenterNet Object Detection Result "
            img_title += "at Step " + str(step+1)
            
            tf_obj_detector.show_object_boxes(
                img_batch[-1], img_bbox[-1], img_dims)
            
            disp_box = tf.constant(disp_box)
            tf_obj_detector.obj_detect_results(
                img_files[-1], voc_model, label_dict, 
                heatmap=True, img_box=disp_box, 
                img_rows=disp_rows, img_cols=disp_cols, 
                img_scale=disp_scale, thresh=thresh, 
                img_title=img_title, save_img_file=save_img_file)
            time.sleep(120)

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
load_pkl_file = tmp_path + "voc_data.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    id_2_label  = pkl.load(tmp_load)
    voc_dataset = pkl.load(tmp_load)

# Define the Neural Network. #
restore_flag = False

disp_rows  = 384
disp_cols  = 384
subsample  = True
n_filters  = 16
step_cool  = 25
init_lr    = 0.001
min_lr     = 1.0e-5
decay_rate = 1.00
max_steps  = 10000
batch_size = 96
sub_batch  = 2
n_classes  = len(id_2_label)
display_step = 25

# Define the checkpoint callback function. #
voc_path = "C:/Users/admin/Desktop/TF_Models/centernet_model/"
train_loss = voc_path + "voc_losses_centernet.csv"
ckpt_model = voc_path + "voc_centernet"

# Load the weights if continuing from a previous checkpoint. #
voc_model = tf_obj_detector.build_model(
    n_filters, n_classes, tmp_pi=0.99, n_features=64, 
    n_repeats=2, seperable=True, batch_norm=True)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    voc_model=voc_model, 
    optimizer=optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

if restore_flag:
    train_loss_df = pd.read_csv(train_loss)
    training_loss = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
    checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        print("Model restored from {}".format(
            ck_manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
else:
    training_loss = []
st_step = checkpoint.step.numpy().astype(np.int32)

# Print out the model summary. #
print(voc_model.summary())
print("-" * 50)

# Subsample the data. #
if subsample:
    train_data = voc_dataset[:100]

print("Fit model on training data (" +\
      str(len(train_data)) + " training samples).")

train(voc_model, n_classes, sub_batch, batch_size, 
      train_data, training_loss, st_step, max_steps, 
      optimizer, checkpoint, ck_manager, id_2_label, 
      disp_rows=disp_rows, disp_cols=disp_cols, 
      display_step=display_step, step_cool=step_cool, 
      init_lr=init_lr, min_lr=min_lr, decay=decay_rate, 
      thresh=1.10, save_flag=True, train_loss_log=train_loss)
print("Model fitted.")
