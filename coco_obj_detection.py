import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import tf_object_detection as tf_obj_detector

from PIL import Image
import matplotlib.pyplot as plt

# Custom function to parse the data. #
def _parse_image(filename):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_width, img_height])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_width, img_height, 3))
    return tf.expand_dims(image_resized, axis=0)

#@tf.function
def train(
    voc_model, sub_batch_sz, batch_size, 
    train_dataset, training_loss, img_disp, img_box, 
    st_step, max_steps, optimizer, ckpt, ck_manager, 
    label_dict, loss_type="focal", init_lr=1.0e-3, 
    decay=0.75, display_step=100, min_lr=1.0e-6, 
    step_cool=50, img_rows=448, img_cols=448, img_scale=None,  
    save_flag=False, save_train_loss_file="train_losses.csv"):
    n_data = len(train_dataset)
    if img_scale is None:
        if max(img_rows, img_cols) >= 512:
            max_scale = max(img_rows, img_cols)
        else:
            max_scale = 512
        img_scale = [64, 128, 256, max_scale]
    else:
        if len(img_scale) != 4:
            raise ValueError("img_scale must be size 4.")
    
    start_time = time.time()
    tot_reg_loss  = 0.0
    tot_cls_loss  = 0.0
    for step in range(st_step, max_steps):
        batch_sample = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        img_files = [train_dataset[x][0] for x in batch_sample]
        img_batch = tf.concat(
            [_parse_image(x) for x in img_files], axis=0)
        
        img_bbox = tf.concat([
            tf.expand_dims(tf.sparse.to_dense(
                tf.sparse.reorder(train_dataset[x][1])), 
                    axis=0) for x in batch_sample], axis=0)
        img_bbox = tf.cast(img_bbox, tf.float32)
        img_mask = img_bbox[:, :, :, :, 4]
        
        epoch = int(step * batch_size / n_data)
        lrate = max(decay**epoch * init_lr, min_lr)
        
        tmp_losses = tf_obj_detector.train_step(
            voc_model, sub_batch_sz, img_batch, 
            img_bbox, img_mask, optimizer, 
            learning_rate=lrate, loss_type=loss_type)
        
        ckpt.step.assign_add(1)
        tot_cls_loss += tmp_losses[0]
        tot_reg_loss += tmp_losses[1]
        if (step+1) % display_step == 0:
            avg_cls_loss = tot_cls_loss.numpy() / display_step
            avg_reg_loss = tot_reg_loss.numpy() / display_step
            training_loss.append((step+1, avg_cls_loss, avg_reg_loss))
            
            print("Average Batch Losses at Step", str(step+1)+":")
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Classification Loss:", str(avg_cls_loss))
            print("Box Regression Loss:", str(avg_reg_loss))
            
            tot_cls_loss = 0.0
            tot_reg_loss = 0.0
            
            elapsed_time = (time.time() - start_time) / 60.0
            print("Elapsed time:", str(elapsed_time), "mins.")
            print("-" * 50)
            
            if (step+1) % step_cool != 0:
                tf_obj_detector.obj_detect_results(
                    img_disp, voc_model, label_dict, 
                    heatmap=True, img_rows=img_rows, img_cols=img_cols, 
                    img_scale=img_scale, img_box=img_box, thresh=0.50)
            start_time = time.time()
        
        if (step+1) % step_cool == 0:
            if save_flag:
                # Save the training losses. #
                train_loss_df = pd.DataFrame(
                    training_loss, columns=["step", "cls_loss", "reg_loss"])
                train_loss_df.to_csv(save_train_loss_file, index=False)
                
                # Save the object detection result. #
                save_img_file = "../Data/Results/COCO_Object_Detection/" +\
                    "object_detection_results_step_" + str(step+1) + ".jpg"
                tf_obj_detector.obj_detect_results(
                    img_disp, voc_model, label_dict, heatmap=True, 
                    img_scale=img_scale, img_rows=img_rows, img_cols=img_cols, 
                    img_box=img_box, thresh=0.50, save_img_file=save_img_file)
                
                # Save the model. #
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            time.sleep(180)

# Load the COCO data. #
print("Loading the file.")
start_time = time.time()

load_pkl_file = "C:/Users/admin/Desktop/Data/COCO/"
load_pkl_file += "coco_annotations_320.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    img_scale = pkl.load(tmp_load)
    coco_obj_list = pkl.load(tmp_load)

# Remember to add 1 more class for background. #
coco_label = pd.read_csv(
    "C:/Users/admin/Desktop/Data/COCO/labels.csv")
list_label = sorted([
    coco_label.iloc[x]["name"] \
    for x in range(len(coco_label))])
list_label = ["background"] + list_label
label_dict = dict([(
    x, list_label[x]) for x in range(len(list_label))])
index_dict = dict([(
    list_label[x], x) for x in range(len(list_label))])

# Define the Neural Network. #
restore_flag = True
sample_flag  = False

#img_width  = 448
#img_height = 448
img_width  = 320
img_height = 320
n_filters  = 32
step_cool  = 125

init_lr    = 1.0e-3
cls_loss   = "focal"
decay_rate = 0.95
max_epochs = 100
batch_size = 96
sub_batch  = 12
max_steps  = 1 +\
    int(max_epochs * len(coco_obj_list) / batch_size)
n_classes  = len(label_dict)

display_step = 25
down_width   = int(img_width/8)
down_height  = int(img_height/8)
dense_shape  = [down_height, down_width, 4, n_classes+4]

# Define the checkpoint callback function. #
coco_path   = "C:/Users/admin/Desktop/TF_Models/coco_model/"
if cls_loss == "focal":
    train_loss  = coco_path + "coco_losses_v0.csv"
    ckpt_path   = coco_path + "coco_v0.ckpt"
    ckpt_dir    = os.path.dirname(ckpt_path)
    ckpt_model  = coco_path + "coco_keras_model_v0"
else:
    train_loss  = coco_path + "coco_losses_sigmoid_v1.csv"
    ckpt_path   = coco_path + "coco_sigmoid_v1.ckpt"
    ckpt_dir    = os.path.dirname(ckpt_path)
    ckpt_model  = coco_path + "coco_keras_model_sigmoid_v1"

# Load the weights if continuing from a previous checkpoint. #
coco_model = tf_obj_detector.build_model(
    n_filters, n_classes, tmp_pi=0.95, 
    img_rows=img_width, img_cols=img_height, seperable=False)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    coco_model=coco_model, 
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
print(coco_model.summary())
print("-" * 50)

elapsed_tm = (time.time() - start_time) / 60
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

print("Fit model on training data (" +\
      str(len(coco_obj_list)) + " training samples).")
img_disp = coco_obj_list[2][0]
img_box  = tf.sparse.to_dense(
    tf.sparse.reorder(coco_obj_list[2][1]))

fig, ax = plt.subplots(1)
tmp_img = np.array(
    Image.open(img_disp), dtype=np.uint8)
ax.imshow(tmp_img)
fig.savefig("test_image.jpg", dpi=199)

plt.close()
del fig, ax

train(coco_model, sub_batch, batch_size, 
      coco_obj_list, training_loss, img_disp, img_box, 
      st_step, max_steps, optimizer, checkpoint, ck_manager, 
      label_dict, loss_type=cls_loss, init_lr=init_lr, 
      decay=decay_rate, step_cool=step_cool, display_step=display_step, 
      img_scale=img_scale, img_rows=img_width, img_cols=img_height, 
      min_lr=1.0e-4, save_flag=True, save_train_loss_file=train_loss)
print("Model fitted.")

save_path = ck_manager.save()
print("Saved model to {}".format(save_path))
