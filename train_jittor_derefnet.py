import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

import jittor as jt
from jittor import optim

from grokcso_jt.models.derefnet_jt import DeRefNetJT
from grokcso_jt.datasets_jt import TrainDatasetJT, ValDatasetJT


def main():
    jt.flags.use_cuda = 1  # 若无 GPU 可改为 0

    # 数据路径与原 DeRefNet 配置保持一致
    train_img_root = os.path.join("SeqCSIST", "data", "track_5000_20", "train", "image")
    train_xml_root = os.path.join("SeqCSIST", "data", "track_5000_20", "train", "annotation")
    val_img_root = os.path.join("SeqCSIST", "data", "track_5000_20", "val", "image")
    val_xml_root = os.path.join("SeqCSIST", "data", "track_5000_20", "val", "annotation")

    train_dataset = TrainDatasetJT(train_img_root, train_xml_root)
    val_dataset = ValDatasetJT(val_img_root, val_xml_root)

    model = DeRefNetJT(layer_no=9)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_epochs = 1  # 可根据需要调大
    seq_len = 20

    num_train_steps = (len(train_dataset) + seq_len - 1) // seq_len
    num_val_steps = (len(val_dataset) + seq_len - 1) // seq_len

    # 训练开始时间（用于计算 ETA）
    train_start_time = time.time()
    
    # 获取当前学习率（Jittor 优化器的学习率）
    current_lr = 1e-4

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_loss_constraint = 0.0
        total_loss_align = 0.0
        total_loss_reg = 0.0
        step = 0
        
        epoch_start_time = time.time()

        for start in range(0, len(train_dataset), seq_len):
            # 数据加载时间
            data_start_time = time.time()
            batch_x, gt_img_11 = train_dataset.get_sequence(start, seq_len)
            data_time = time.time() - data_start_time

            # 迭代时间
            iter_start_time = time.time()
            
            loss_dict = model(batch_x=batch_x, gt_img_11=gt_img_11)
            loss = loss_dict["loss"]

            opt.step(loss)
            
            iter_time = time.time() - iter_start_time

            # 累计损失
            total_loss += float(loss.data)
            total_loss_constraint += float(loss_dict["loss_constraint"].data)
            total_loss_align += float(loss_dict["loss_align"].data)
            total_loss_reg += float(loss_dict["loss_reg"].data)
            step += 1

            # 计算 ETA
            elapsed_time = time.time() - train_start_time
            avg_time_per_iter = elapsed_time / (epoch * num_train_steps + step)
            remaining_iters = (max_epochs - epoch - 1) * num_train_steps + (num_train_steps - step)
            eta_seconds = avg_time_per_iter * remaining_iters

            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            seconds = int(eta_seconds % 60)
            eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # 格式化时间戳
            timestamp = datetime.now().strftime("%m/%d %H:%M:%S")

            print(
                f"{timestamp} - mmengine - INFO - Epoch(train)   "
                f"[{epoch+1}][{step:3d}/{num_train_steps}]  "
                f"lr: {current_lr:.4e}  "
                f"eta: {eta_str}  "
                f"time: {iter_time:.4f}  "
                f"data_time: {data_time:.4f}  "
                f"loss: {float(loss.data):.4f}  "
                f"loss_constraint: {float(loss_dict['loss_constraint'].data):.4f}  "
                f"loss_align: {float(loss_dict['loss_align'].data):.4f}  "
                f"loss_reg: {float(loss_dict['loss_reg'].data):.4f}"
            )

        avg_train_loss = total_loss / max(step, 1)
        avg_loss_constraint = total_loss_constraint / max(step, 1)
        avg_loss_align = total_loss_align / max(step, 1)
        avg_loss_reg = total_loss_reg / max(step, 1)
        
        epoch_time = time.time() - epoch_start_time
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        print(
            f"{timestamp} - mmengine - INFO - Epoch(train)   "
            f"[{epoch+1}]  "
            f"avg_loss: {avg_train_loss:.4f}  "
            f"avg_loss_constraint: {avg_loss_constraint:.4f}  "
            f"avg_loss_align: {avg_loss_align:.4f}  "
            f"avg_loss_reg: {avg_loss_reg:.4f}  "
            f"time: {epoch_time:.2f}s\n"
        )

        # 验证
        model.eval()
        val_start_time = time.time()
        with jt.no_grad():
            val_loss = 0.0
            val_loss_constraint = 0.0
            val_loss_align = 0.0
            val_loss_reg = 0.0
            v_step = 0
            for start in range(0, len(val_dataset), seq_len):
                data_start_time = time.time()
                batch_x, gt_img_11 = val_dataset.get_sequence(start, seq_len)
                data_time = time.time() - data_start_time
                
                iter_start_time = time.time()
                loss_dict = model(batch_x=batch_x, gt_img_11=gt_img_11)
                iter_time = time.time() - iter_start_time
                
                loss_val = float(loss_dict["loss"].data)
                val_loss += loss_val
                val_loss_constraint += float(loss_dict["loss_constraint"].data)
                val_loss_align += float(loss_dict["loss_align"].data)
                val_loss_reg += float(loss_dict["loss_reg"].data)
                v_step += 1

                timestamp = datetime.now().strftime("%m/%d %H:%M:%S")

                print(
                    f"{timestamp} - mmengine - INFO - Epoch(val)     "
                    f"[{epoch+1}][{v_step:3d}/{num_val_steps}]  "
                    f"time: {iter_time:.4f}  "
                    f"data_time: {data_time:.4f}  "
                    f"loss: {loss_val:.4f}  "
                    f"loss_constraint: {float(loss_dict['loss_constraint'].data):.4f}  "
                    f"loss_align: {float(loss_dict['loss_align'].data):.4f}  "
                    f"loss_reg: {float(loss_dict['loss_reg'].data):.4f}"
                )

        avg_val_loss = val_loss / max(v_step, 1)
        avg_val_loss_constraint = val_loss_constraint / max(v_step, 1)
        avg_val_loss_align = val_loss_align / max(v_step, 1)
        avg_val_loss_reg = val_loss_reg / max(v_step, 1)
        val_time = time.time() - val_start_time
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        print(
            f"{timestamp} - mmengine - INFO - Epoch(val)     "
            f"[{epoch+1}]  "
            f"avg_loss: {avg_val_loss:.4f}  "
            f"avg_loss_constraint: {avg_val_loss_constraint:.4f}  "
            f"avg_loss_align: {avg_val_loss_align:.4f}  "
            f"avg_loss_reg: {avg_val_loss_reg:.4f}  "
            f"time: {val_time:.2f}s"
        )
        print("--------------------------------------------------------\n")

    # 保存 Jittor 权重
    save_path = os.path.join("work_dir", "DeRefNet_jittor")
    os.makedirs(save_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, "final_model.pkl")
    jt.save(model.state_dict(), ckpt_path)
    print(f"========== train end，模型已保存到: {ckpt_path} ==========")


if __name__ == "__main__":
    main()


