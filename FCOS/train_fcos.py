
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from data_preprocess import resize_image
from fcos import build_model, format_data, model_loss

# Training function. #
def train(
    train_data, training_loss, model, 
    batch_size, optimizer, ckpt, ck_manager, 
    st_step, max_steps, init_lr=1.0e-3, min_lr=1.0e-5, 
    decay_step=1000, display_step=50, step_save=100, 
    step_cool=1000, save_loss_file="train_losses.csv"):
    n_data = len(train_data)
    strides = [8, 16, 32, 64, 128]
    tmp_sizes = [32, 64, 128, 256]
    
    model_params  = model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    start_time = time.time()
    batch_objs = 0
    total_loss = 0.0
    trend_loss = 0.0
    
    tot_reg_loss = 0.0
    tot_cls_loss = 0.0
    tot_cen_loss = 0.0
    for step in range(st_step, max_steps):
        step_lr = max(init_lr * np.power(
            0.5, int(step / decay_step)), min_lr)
        optimizer.lr.assign(step_lr)
        
        batch_sample = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        # FCOS loss is computed per image. #
        num_object = 0
        acc_losses = 0.0
        reg_losses = 0.0
        cls_losses = 0.0
        cen_losses = 0.0
        for tmp_idx in batch_sample:
            image, bbox, class_id = \
                resize_image(train_data[tmp_idx])
            class_id = tf.cast(class_id, tf.float32)
            
            label = tf.concat([
                bbox, tf.expand_dims(class_id, 1)], axis=1)
            image = tf.expand_dims(image, axis=0)
            
            img_dim = [int(image.shape[1]), 
                       int(image.shape[2])]
            
            tmp_labels, n_labels = format_data(
                label, img_dim, num_classes)
            with tf.GradientTape() as grad_tape:
                tmp_output = model(image, training=True)
                tmp_losses = model_loss(
                    tmp_labels, tmp_output, 
                    strides, cls_lambda=1.0)
                all_losses = \
                    tmp_losses[0] + tmp_losses[1] + tmp_losses[2]
            
            # Accumulate the gradients. #
            num_object += sum(n_labels)
            cls_losses += tmp_losses[0]
            reg_losses += tmp_losses[1]
            cen_losses += tmp_losses[2]
            acc_losses += all_losses
            
            tmp_gradients = \
                grad_tape.gradient(all_losses, model_params)
            acc_gradients = [(acc_grad + grad) for \
                acc_grad, grad in zip(acc_gradients, tmp_gradients)]
        
        # Update the weights. #
        acc_gradients = [tf.math.divide_no_nan(
            acc_grad, batch_size) for acc_grad in acc_gradients]
        optimizer.apply_gradients(
            zip(acc_gradients, model_params))
        
        ckpt.step.assign_add(1)
        batch_objs += num_object / batch_size
        total_loss += acc_losses / batch_size
        trend_loss += acc_losses / batch_size
        
        tot_reg_loss += reg_losses / batch_size
        tot_cls_loss += cls_losses / batch_size
        tot_cen_loss += cen_losses / batch_size
        if (step+1) % display_step == 0:
            avg_loss = total_loss / display_step
            avg_objs = batch_objs / display_step
            
            avg_reg_loss = tot_reg_loss / display_step
            avg_cls_loss = tot_cls_loss / display_step
            avg_cen_loss = tot_cen_loss / display_step
            avg_reg_loss = avg_reg_loss.numpy()
            avg_cls_loss = avg_cls_loss.numpy()
            avg_cen_loss = avg_cen_loss.numpy()
            
            batch_objs = 0
            total_loss = 0.0
            tot_reg_loss = 0.0
            tot_cls_loss = 0.0
            tot_cen_loss = 0.0
            
            elapsed_tm = (time.time() - start_time) / 60.0
            start_time = time.time()
            
            print("Iteration:", str(step+1))
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Average Objs:", str(avg_objs))
            print("Average Loss:", str(round(avg_loss.numpy(), 5)))
            print("Average Reg Loss:", str(round(avg_reg_loss, 5)))
            print("Average Cls Loss:", str(round(avg_cls_loss, 5)))
            print("Average Cen Loss:", str(round(avg_cen_loss, 5)))
            
            if (step+1) % step_save == 0:
                # Save the training losses. #
                training_loss.append((step+1, avg_loss))
                
                df_columns = ["step", "train_loss"]
                train_loss_df = pd.DataFrame(
                    training_loss, columns=df_columns)
                train_loss_df.to_csv(save_loss_file, index=False)
                
                # Save the model. #
                print("")
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            
            if (step+1) % step_cool != 0:
                print("Elapsed Time:", str(elapsed_tm), "mins.")
                print("-" * 50)
        
        if (step+1) % step_cool == 0:
            avg_trend = trend_loss.numpy() / step_cool
            trend_loss = 0.0
            
            print("Trend Loss:", str(round(avg_trend, 5)))
            print("Elapsed Time:", str(elapsed_tm), "mins.")
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
voc_dataset = voc_dataset[:100]

# Define the checkpoint callback function. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
train_loss = voc_path + "voc_losses_fcos_mobilenetv2.csv"
ckpt_model = voc_path + "voc_fcos_mobilenetv2"
restore_flag = True

# Training Parameters. #
init_lr = 5.0e-4
step_cool  = 125
max_steps  = 500
disp_step  = 25
decay_step = 10000

batch_size = 16
sub_batch  = 4
num_classes = len(id_2_label)

fcos_model = build_model(
    num_classes, backbone_model="mobilenetv2")
model_optimizer = tf.optimizers.Adam()
#model_optimizer = tf.optimizers.SGD(
#    learning_rate=1.0e-3, momentum=0.9)

print(fcos_model.summary())

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    fcos_model=fcos_model, 
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
print("")
print("Training FoveaNet Model with", str(num_classes), 
      "classes (" + str(st_step) + " iterations).")

train(
    voc_dataset, training_loss, 
    fcos_model, batch_size, model_optimizer, 
    checkpoint, 
    ck_manager, st_step, max_steps, init_lr=init_lr, 
    decay_step=decay_step, display_step=disp_step, 
    step_cool=step_cool, save_loss_file=train_loss)
