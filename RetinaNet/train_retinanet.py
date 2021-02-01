
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from data_preprocess import LabelEncoder, preprocess_data
from retinanet import get_backbone, RetinaNet, RetinaNetLoss

# Training function. #
def train(
    train_data, training_loss, model, 
    label_encoder, loss_fn, batch_size, 
    sub_batch_sz, optimizer, ckpt, ck_manager, 
    st_step, max_steps, display_step=10, 
    step_cool=250, save_loss_file="train_losses.csv"):
    n_data = len(train_data)
    
    model_params  = model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    start_time = time.time()
    total_loss = 0.0
    for step in range(st_step, max_steps):
        #optimizer.lr.assign(learning_rate)
        batch_sample = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        tmp_images = []
        tmp_bboxes = []
        tmp_labels = []
        batch_stats = []
        for tmp_idx in batch_sample:
            image, bbox, class_id = \
                preprocess_data(train_data[tmp_idx])
            
            image_shp  = image.shape
            img_width  = int(image_shp[0])
            img_height = int(image_shp[1])
            n_bbox = int(tf.shape(bbox)[0])
            
            tmp_bboxes.append(bbox)
            tmp_images.append(image)
            tmp_labels.append(class_id)
            batch_stats.append((img_width, img_height, n_bbox))
        
        # Get the padding statistics. #
        max_bbox = max([z for x, y, z in batch_stats])
        max_width  = max([x for x, y, z in batch_stats])
        max_height = max([y for x, y, z in batch_stats])
        
        img_batch = []
        img_bbox  = []
        img_label = []
        for n_sample in range(batch_size):
            batch_stat = batch_stats[n_sample]
            
            n_zero = max_bbox - batch_stat[2]
            bbox_pad  = 1.0e-8 * tf.ones([n_zero, 4])
            label_pad = -1 * tf.ones([n_zero,], dtype=tf.int32)
            
            img_bbox.append(tf.expand_dims(tf.concat([
                tmp_bboxes[n_sample], bbox_pad], axis=0), axis=0))
            img_label.append(tf.expand_dims(tf.concat([
                tmp_labels[n_sample], label_pad], axis=0), axis=0))
            img_batch.append(tf.expand_dims(
                tf.image.resize_with_crop_or_pad(
                    tmp_images[n_sample], 
                    max_width, max_height), axis=0))
        del tmp_images, tmp_bboxes, tmp_labels
        
        img_batch = tf.concat(img_batch, axis=0)
        img_bbox  = tf.concat(img_bbox, axis=0)
        img_label = tf.concat(img_label, axis=0)
        
        batch_images, labels = \
            label_encoder.encode_batch(
                img_batch, img_bbox, img_label)
        
        if batch_size <= sub_batch_sz:
            n_sub_batch = 1
        elif batch_size % sub_batch_sz == 0:
            n_sub_batch = int(batch_size / sub_batch_sz)
        else:
            n_sub_batch = int(batch_size / sub_batch_sz) + 1
        
        acc_losses = 0.0
        for n_sub in range(n_sub_batch):
            id_st = n_sub*sub_batch_sz
            if n_sub != (n_sub_batch-1):
                id_en = (n_sub+1)*sub_batch_sz
            else:
                id_en = batch_size
            
            with tf.GradientTape() as grad_tape:
                tmp_output = model(
                    batch_images[id_st:id_en], training=True)
                tmp_losses = loss_fn(
                    labels[id_st:id_en], tmp_output)
            
            # Accumulate the gradients. #
            acc_losses += tmp_losses
            tmp_gradients = \
                grad_tape.gradient(tmp_losses, model_params)
            acc_gradients = [(acc_grad + grad) for \
                acc_grad, grad in zip(acc_gradients, tmp_gradients)]
        
        # Update the weights. #
        acc_gradients = [tf.math.divide_no_nan(
            acc_grad, batch_size) for acc_grad in acc_gradients]
        optimizer.apply_gradients(
            zip(acc_gradients, model_params))
        
        ckpt.step.assign_add(1)
        total_loss += acc_losses
        if (step+1) % display_step == 0:
            avg_loss = total_loss.numpy() / display_step
            total_loss = 0.0
            
            elapsed_tm = (time.time() - start_time) / 60.0
            start_time = time.time()
            
            print("Iteration:", str(step+1))
            print("Average Loss:", str(avg_loss))
            print("Elapsed Time:", str(elapsed_tm), "mins.")
            print("-" * 50)
        
        if (step+1) % step_cool == 0:
            # Save the training losses. #
            training_loss.append((step+1, avg_loss))
            
            df_columns = ["step", "train_loss"]
            train_loss_df = pd.DataFrame(
                training_loss, columns=df_columns)
            train_loss_df.to_csv(save_loss_file, index=False)
            
            # Save the model. #
            save_path = ck_manager.save()
            print("Saved model to {}".format(save_path))
            
            print("Cooling GPU for 2 minutes.")
            time.sleep(120)
            print("-" * 50)
    return None

# Load the data. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
load_pkl_file = tmp_path + "voc_data.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    id_2_label  = pkl.load(tmp_load)
    voc_dataset = pkl.load(tmp_load)

# Define the checkpoint callback function. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
train_loss = voc_path + "voc_losses_retinanet_mobilenetv2.csv"
ckpt_model = voc_path + "voc_retinanet_mobilenetv2"
restore_flag = True

# Training Parameters. #
step_cool = 500
max_steps = 10000
disp_step = 50
backbone_name = "mobilenetv2"
if backbone_name == "mobilenetv2":
    preprocess_input = \
        tf.keras.applications.mobilenet_v2.preprocess_input
else:
    preprocess_input = None

box_dims = [float(300 / (2**x)) for x in range(5)]
box_dims = box_dims[::-1]
label_encoder = LabelEncoder(
    preprocess_input=preprocess_input, box_dims=box_dims)

batch_size = 8
sub_batch  = 4
num_classes = len(id_2_label)

loss_function = RetinaNetLoss(num_classes)
mobilenet_backbone = get_backbone(model_name=backbone_name)

retinanet_model = RetinaNet(
    num_classes, mobilenet_backbone)
model_optimizer = tf.optimizers.SGD(
    learning_rate=1.0e-3, momentum=0.9)

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    retinanet_model=retinanet_model, 
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

# Training the model. #
train(
    voc_dataset, training_loss, 
    retinanet_model, label_encoder, 
    loss_function, batch_size, sub_batch, 
    model_optimizer, checkpoint, ck_manager, 
    st_step, max_steps, display_step=disp_step, 
    step_cool=step_cool, save_loss_file=train_loss)
