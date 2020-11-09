import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import tf_spine_net as tf_obj_detector

from PIL import Image
import matplotlib.pyplot as plt

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
        if p_tmp <= 0.333:
            # Distort colour. #
            tmp_in = img_in.numpy()
            
            tmp_in[:, :, 0] = 1.0 - tmp_in[:, :, 0]
            tmp_in[:, :, 1] = 1.0 - tmp_in[:, :, 1]
            tmp_in[:, :, 2] = 1.0 - tmp_in[:, :, 2]
            return (tf.constant(tmp_in), img_bbox)
        elif p_tmp <= 0.667:
            # Flip left-right. #
            img_in = img_in[:, ::-1, :]
            
            tmp_bbox = img_bbox.numpy()
            img_bbox = tmp_bbox[:, ::-1, :, :]
            
            img_bbox[:, :, :, 1] = 1.0 - img_bbox[:, :, :, 1]
            return (img_in, tf.constant(img_bbox))
        else:
            # Rotate by either 90 degrees of 270 degrees. #
            img_in = tf.transpose(img_in, [1, 0, 2])
            
            img_bbox = tf.transpose(img_bbox, [1, 0, 2, 3])
            tmp_bbox = img_bbox.numpy()
            img_bbox = tmp_bbox
            
            img_bbox[:, :, :, 0] = tmp_bbox[:, :, :, 1]
            img_bbox[:, :, :, 1] = tmp_bbox[:, :, :, 0]
            img_bbox[:, :, :, 2] = tmp_bbox[:, :, :, 3]
            img_bbox[:, :, :, 3] = tmp_bbox[:, :, :, 2]
            
            p_rot = np.random.uniform()
            if p_rot >= 0.50:
                # Rotate by 270 degrees by flipping up-down. #
                img_in = img_in[::-1, :, :]
                
                img_bbox = img_bbox[::-1, :, :, :]
                img_bbox[:, :, :, 0] = 1.0 - img_bbox[:, :, :, 0]
            
            img_bbox = tf.constant(img_bbox)
            return (img_in, img_bbox)
    else:
        return (img_in, img_bbox)

def train(
    voc_model, sub_batch_sz, batch_size, 
    train_dataset, training_loss, img_disp, img_box, 
    st_step, max_steps, optimizer, ckpt, ck_manager, 
    label_dict, decay=0.75, init_lr=1.0e-3, img_scale=None,
    disp_rows=320, disp_cols=320, disp_scale=None, 
    display_step=100, step_cool=50, min_lr=1.0e-6, 
    save_flag=False, save_train_loss_file="train_losses.csv"):
    n_data  = min([len(
        train_dataset[x]) for x \
        in range(len(train_dataset))])
    #n_scales = len(train_dataset)
    
    start_time = time.time()
    tot_reg_loss  = 0.0
    tot_cls_loss  = 0.0
    for step in range(st_step, max_steps):
        #batch_scale = np.random.choice(n_scales, size=1)[0]
        batch_scale = 0
        
        tmp_dims = train_dataset[batch_scale][0][1]
        img_rows = tmp_dims[0]
        img_cols = tmp_dims[1]
        tmp_data = train_dataset[batch_scale]
        
        min_scale = min(tmp_dims)
        if img_scale is None:
            tmp_scale = [int(
                min_scale/(2**x)) for x in range(4)]
        else:
            tmp_scale = img_scale[batch_scale]
            if len(tmp_scale) != 4:
                raise ValueError("img_scale must be size 4.")
        
        batch_sample = np.random.choice(
            len(tmp_data), size=batch_size, replace=False)
        
        img_files = [tmp_data[x][0] for x in batch_sample]
        img_batch = [_parse_image(
            x, img_rows=img_rows, 
            img_cols=img_cols) for x in img_files]
        img_bbox  = [
            tf.sparse.to_dense(tf.sparse.reorder(
                tmp_data[x][2])) for x in batch_sample]
        
        img_tuple = [image_augment(
            img_batch[x], img_bbox[x]) \
                for x in range(batch_size)]
        img_batch = [tf.expand_dims(
            x, axis=0) for x, y in img_tuple]
        img_bbox  = [tf.expand_dims(
            y, axis=0) for x, y in img_tuple]
        
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
                img_title = "Object Detection Result "
                img_title += "at Step " + str(step+1)
                
                tf_obj_detector.obj_detect_results(
                    img_disp, voc_model, 
                    label_dict, heatmap=True, 
                    img_box=img_box, thresh=0.50, 
                    img_rows=disp_rows, img_cols=disp_cols, 
                    img_scale=disp_scale, img_title=img_title)
                print("-" * 50)
        
        if (step+1) % step_cool == 0:
            if save_flag:
                # Save the training losses. #
                train_loss_df = pd.DataFrame(
                    training_loss, columns=["step", "cls_loss", "reg_loss"])
                train_loss_df.to_csv(save_train_loss_file, index=False)
                
                # Save the model. #
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            print("-" * 50)
            
            save_img_file = "C:/Users/admin/Desktop/Data/"
            save_img_file += "Results/iMat_Object_Detection/"
            save_img_file += "iMat_obj_detect_" + str(step+1) + ".jpg"
            
            img_title = "Object Detection Result "
            img_title += "at Step " + str(step+1)
            
            tf_obj_detector.obj_detect_results(
                img_disp, voc_model, label_dict, 
                heatmap=True, img_box=img_box, 
                img_scale=disp_scale, thresh=0.50, 
                img_rows=disp_rows, img_cols=disp_cols, 
                img_title=img_title, save_img_file=save_img_file)
            time.sleep(120)

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/iMaterialist/"
load_pkl_file = tmp_path + "iMat_annotations_multiscale.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    img_scale = pkl.load(tmp_load)
    label_dict = pkl.load(tmp_load)
    iMat_object_list = pkl.load(tmp_load)

# Define the Neural Network. #
# Add in the objectness class to n_classes. #
restore_flag = True

n_filters  = 24
step_cool  = 50
init_lr    = 1.0e-3
decay_rate = 0.975
max_steps  = 25000
batch_size = 96
sub_batch  = 6
n_classes  = len(label_dict) + 1
display_step = 25

# Define the checkpoint callback function. #
iMat_path = "C:/Users/admin/Desktop/TF_Models/iMat_model/"
train_loss  = iMat_path + "iMat_losses_v0.csv"
ckpt_model  = iMat_path + "iMat_keras_model_v0"

# Load the weights if continuing from a previous checkpoint. #
iMat_model = tf_obj_detector.build_model(
    n_filters, n_classes, tmp_pi=0.95, 
    n_repeats=3, seperable=True, batch_norm=True)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    iMat_model=iMat_model, 
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
print(iMat_model.summary())
print("-" * 50)

print("Fit model on training data (" +\
      str(len(iMat_object_list[0])) + " training samples).")
img_disp = iMat_object_list[0][10][0]
img_dims = iMat_object_list[0][10][1]
img_box  = tf.cast(
    tf.sparse.to_dense(tf.sparse.reorder(
        iMat_object_list[0][10][2])), tf.float32)
disp_scale = img_scale[0]

fig, ax = plt.subplots(1)
tmp_img = np.array(
    Image.open(img_disp), dtype=np.uint8)
ax.imshow(tmp_img)
fig.savefig("test_image.jpg", dpi=199)

plt.close()
del fig, ax

train(iMat_model, sub_batch, batch_size, 
      iMat_object_list, training_loss, 
      img_disp, img_box, st_step, max_steps, optimizer, 
      checkpoint, ck_manager, label_dict, init_lr=init_lr, min_lr=1.0e-5, 
      decay=decay_rate, step_cool=step_cool, display_step=display_step, 
      img_scale=img_scale, disp_rows=img_dims[0], disp_cols=img_dims[1], 
      disp_scale=disp_scale, save_flag=True, save_train_loss_file=train_loss)
print("Model fitted.")

save_path = ck_manager.save()
print("Saved model to {}".format(save_path))
