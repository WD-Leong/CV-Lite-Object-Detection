import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_centernet_resnet_s8 as tf_obj_detector
from data_preprocess import random_flip_horizontal
from data_preprocess import swap_xy, convert_to_xywh

# Custom function to parse the data. #
def _parse_image(
    filename, img_rows=448, img_cols=448):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = image_decoded / 255.0
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized ,shape=(img_rows, img_cols, 3))
    return image_resized

def train(
    model, n_classes, img_dims, 
    sub_batch_sz, batch_size, box_scales, 
    train_data, training_loss, st_step, max_steps, optimizer, 
    ckpt, ck_manager, label_dict, init_lr=1.0e-3, min_lr=1e-6, 
    downsample=32, use_scale=False, min_scale=0.7, decay=0.75, 
    display_step=100, step_cool=50, base_rows=320, base_cols=320, 
    thresh=0.50, save_flag=False, train_loss_log="train_losses.csv"):
    n_data = len(train_data)
    base_dims = min(base_rows, base_cols)
    max_scale = img_dims / base_dims
    
    start_time = time.time()
    tot_reg_loss  = 0.0
    tot_cls_loss  = 0.0
    for step in range(st_step, max_steps):
        if step < 20000:
            lrate = init_lr
        elif step < 25000:
            lrate = init_lr / 10.0
        else:
            lrate = init_lr / 100.0
        lrate = max(lrate, min_lr)
        
        batch_sample = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        # Use only one image resolution to train. #
        if use_scale:
            rnd_scale = np.random.uniform(
                low=min_scale, high=max_scale)
        else:
            rnd_scale = max_scale
        
        raw_dims = int(rnd_scale * base_dims)
        pad_dims = int((img_dims - raw_dims) / 2.0)
        sc_dims  = [raw_dims, raw_dims]
        img_pad  = [img_dims, img_dims]
        
        img_boxes = []
        img_batch = []
        for tmp_idx in batch_sample:
            tmp_image = _parse_image(
                train_data[tmp_idx]["image"], 
                img_rows=raw_dims, img_cols=raw_dims)
            tmp_bbox  = np.array(train_data[
                tmp_idx]["objects"]["bbox"])
            tmp_class = np.array(train_data[
                tmp_idx]["objects"]["label"])
            tmp_class = np.expand_dims(tmp_class, axis=1)
            
            disp_bbox = swap_xy(tmp_bbox)
            disp_bbox = convert_to_xywh(disp_bbox)
            disp_label = np.concatenate(tuple([
                disp_bbox, tmp_class]), axis=1)
            disp_label = tf.constant(disp_label)
            del disp_bbox
            
            disp_tuple = tf_obj_detector.format_data(
                disp_label, box_scales, img_pad, 
                n_classes, img_pad=img_pad, stride=8)
            
            tmp_tuple = \
                random_flip_horizontal(tmp_image, tmp_bbox)
            tmp_image = tmp_tuple[0]
            
            tmp_bbox = tmp_tuple[1]
            tmp_bbox = swap_xy(tmp_bbox)
            tmp_bbox = convert_to_xywh(tmp_bbox)
            
            tmp_image = tf.image.pad_to_bounding_box(
                tmp_image, pad_dims, pad_dims, img_dims, img_dims)
            gt_labels = np.concatenate(tuple([
                tmp_bbox, tmp_class]), axis=1)
            gt_labels = tf.constant(gt_labels)
            del tmp_tuple
            
            tmp_tuple = tf_obj_detector.format_data(
                gt_labels, box_scales, sc_dims, 
                n_classes, img_pad=img_pad, stride=8)
            
            img_batch.append(
                tf.expand_dims(tmp_image, axis=0))
            img_boxes.append(
                tf.expand_dims(tmp_tuple[0], axis=0))
            del tmp_tuple, gt_labels, tmp_image, tmp_bbox
        
        # Get the image file names. #
        img_files = [
            train_data[x]["image"] for x in batch_sample]
        
        # Note that TF parses the image transposed, so the  #
        # bounding boxes coordinates are already transposed #
        # during the formatting of the data.                #
        img_batch = tf.concat(img_batch, axis=0)
        img_boxes = tf.cast(tf.concat(
            img_boxes, axis=0), tf.float32)
        
        tmp_losses = tf_obj_detector.train_step(
            model, sub_batch_sz, img_batch, 
            img_boxes, optimizer, learning_rate=lrate)
        
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
                img_title = "CenterNetv1 ResNet-101 "
                img_title += "Object Detection Result "
                img_title += "at Step " + str(step+1)
                
                tmp_img  = img_batch[-1]
                tmp_bbox = img_boxes[-1]
                tf_obj_detector.show_object_boxes(
                    tmp_img, tmp_bbox, img_dims, 
                    box_scales, downsample=downsample)
                
                tf_obj_detector.obj_detect_results(
                    img_files[-1], model, 
                    box_scales, label_dict, heatmap=True, 
                    thresh=thresh, downsample=downsample, 
                    img_rows=img_dims, img_cols=img_dims, 
                    img_box=disp_tuple[0], img_title=img_title)
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
            
            img_title = "CenterNetv1 ResNet-101 "
            img_title += "Object Detection Result "
            img_title += "at Step " + str(step+1)
            
            tmp_img  = img_batch[-1]
            tmp_bbox = img_boxes[-1]
            tf_obj_detector.show_object_boxes(
                tmp_img, tmp_bbox, img_dims, 
                box_scales, downsample=downsample)
            
            tf_obj_detector.obj_detect_results(
                img_files[-1], model, 
                box_scales, label_dict, heatmap=True, 
                thresh=thresh, downsample=downsample, 
                img_rows=img_dims, img_cols=img_dims, 
                img_box=disp_tuple[0], img_title=img_title)
            time.sleep(120)

# Load the Crowd Human dataset. #
tmp_path  = "C:/Users/admin/Desktop/Data/Crowd Human Dataset/"
data_file = tmp_path + "crowd_human_body_data.pkl"
with open(data_file, "rb") as tmp_load:
    train_data = pkl.load(tmp_load)

# Generate the label dictionary. #
id_2_label = dict([(0, "person")])

# Define the Neural Network. #
restore_flag = False

subsample  = False
downsample = 8
img_dims   = 512
base_rows  = 448
base_cols  = 448
disp_rows  = img_dims
disp_cols  = img_dims
step_cool  = 50
init_lr    = 0.01
min_lr     = 1.0e-5
decay_rate = 0.999
max_steps  = 30000
batch_size = 16
sub_batch  = 1
n_classes  = len(id_2_label)
box_scales = [32.0, 64.0, 128.0, 256.0, 512.0]
n_scales   = len(box_scales)
display_step = 25

if subsample:
    train_data = train_data[:2500]

# Define the checkpoint callback function. #
model_path = \
    "C:/Users/admin/Desktop/TF_Models/crowd_human_model/"
train_loss = \
    model_path + "crowd_human_losses_centernet_resnet101.csv"
ckpt_model = model_path + "crowd_human_centernet_resnet101"

# Build the model. #
centernet_model = tf_obj_detector.build_model(
    n_classes, n_scales=n_scales, backbone_model="resnet101")
model_optimizer = tf.keras.optimizers.SGD(momentum=0.9)

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    centernet_model=centernet_model, 
    model_optimizer=model_optimizer)
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
print(centernet_model.summary())
print("-" * 50)

print("Fit model on training data (" +\
      str(len(train_data)) + " training samples).")

train(centernet_model, n_classes, img_dims, 
      sub_batch, batch_size, box_scales, 
      train_data, training_loss, st_step, 
      max_steps, model_optimizer, checkpoint, 
      ck_manager, id_2_label, decay=decay_rate, 
      base_rows=base_rows, base_cols=base_cols, 
      display_step=display_step, step_cool=step_cool, 
      init_lr=init_lr, min_lr=min_lr, downsample=downsample, 
      thresh=0.50, save_flag=True, train_loss_log=train_loss)
print("Model fitted.")
