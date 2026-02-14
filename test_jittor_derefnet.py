import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import jittor as jt

from grokcso_jt.models.derefnet_jt import DeRefNetJT
from grokcso_jt.datasets_jt import ValDatasetJT


def parse_args():
    parser = argparse.ArgumentParser(description="Test DeRefNet with Jittor (single GPU)")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("work_dir", "DeRefNet_jittor", "final_model.pkl"),
        help="Jittor checkpoint 路径（默认使用 train_jittor_derefnet.py 训练得到的权重）",
    )
    parser.add_argument(
        "--use_cuda",
        type=int,
        default=1,
        help="是否使用 GPU，1 表示使用，0 表示只用 CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    jt.flags.use_cuda = args.use_cuda

    # 使用 test 集进行测试，对应原 DeRefNet 配置
    test_img_root = os.path.join("SeqCSIST", "data", "track_5000_20", "test", "image")
    test_xml_root = os.path.join("SeqCSIST", "data", "track_5000_20", "test", "annotation")

    test_dataset = ValDatasetJT(test_img_root, test_xml_root)

    model = DeRefNetJT(layer_no=9)

    print("========== Jittor DeRefNet test begin ==========")
    print(f"Device: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    print(f"Test samples: {len(test_dataset)}")

    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"加载 Jittor 权重: {args.ckpt}")
        state_dict = jt.load(args.ckpt)
        model.load_parameters(state_dict)
    else:
        print(f"未找到指定权重 {args.ckpt}，将使用随机初始化模型进行测试。")

    model.eval()

    seq_len = 20
    num_steps = (len(test_dataset) + seq_len - 1) // seq_len
    total_loss = 0.0
    total_samples = 0

    with jt.no_grad():
        step = 0
        for start in range(0, len(test_dataset), seq_len):
            batch_x, gt_img_11 = test_dataset.get_sequence(start, seq_len)
            loss_dict = model(batch_x=batch_x, gt_img_11=gt_img_11)
            loss = float(loss_dict["loss"].data)

            step += 1
            total_loss += loss
            total_samples += 1

            # 测试阶段每个 batch 的日志
            print(
                f"[Test] Iter[{step}/{num_steps}]  "
                f"loss: {loss:.6f}"
            )

    avg_loss = total_loss / max(total_samples, 1)
    print(f"[Test] Average loss: {avg_loss:.6f}")
    print("========== test end ==========")


if __name__ == "__main__":
    main()


