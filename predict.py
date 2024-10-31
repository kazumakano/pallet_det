import csv
import os
import os.path as path
from glob import glob, iglob
from typing import Optional
import cv2
import pandas as pd
import ray
from pycocotools import mask as coco_mask
from torch import cuda
from torchvision.transforms import functional as TF
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results as YoloResults

GPU_PER_TASK = 0.5

@ray.remote(num_gpus=GPU_PER_TASK)
def _predict_by_file(model_file: str, result_dir: str, vid_file: str) -> None:
    cap = cv2.VideoCapture(filename=vid_file)
    model = YOLO(model=model_file)

    with open(path.join(result_dir, "pallet_results_v2.csv"), mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(("id", "x", "y", "w", "h"))

        for _, r in pd.read_csv(path.join(result_dir, "segment_results.csv")).iterrows():
            while r["frm_idx"] >= cap.get(cv2.CAP_PROP_POS_FRAMES):
                frm = cap.read()[1]
            masked_frm = cv2.bitwise_and(frm, frm, mask=255 * coco_mask.decode({"counts": r["mask"], "size": (1080, 1920)}))

            results: YoloResults = model(masked_frm)[0]

            for b in results.boxes:
                writer.writerow((r["id"], b.xywh[0, 0].item(), b.xywh[0, 1].item(), b.xywh[0, 2].item(), b.xywh[0, 3].item()))

    cap.release()

def predict(model_file: str, result_dir: str, vid_dir: str, gpu_ids: Optional[list[int]] = None) -> None:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])
    ray.init()

    pid_queue = []
    for cn in sorted(os.listdir(result_dir)):
        if path.isdir(cd := path.join(result_dir, cn)):
            for vd in tqdm(sorted(iglob(path.join(cd, "??-??-??_*"))), desc=f"predicting camera {cn}"):
                if len(pid_queue) >= cuda.device_count() // GPU_PER_TASK:
                    pid_queue.remove(ray.wait(pid_queue, num_returns=1)[0][0])
                pid_queue.append(_predict_by_file.remote(path.abspath(model_file), path.abspath(vd), path.abspath(glob(path.join(vid_dir, f"camera{cn}/video_{path.basename(vd)}.mp4"))[0])))

    ray.get(pid_queue)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", required=True, help="specify model file", metavar="PATH_TO_MODEL_FILE")
    parser.add_argument("-r", "--result_dir", required=True, help="specify segment result directory", metavar="PATH_TO_RESULT_DIR")
    parser.add_argument("-v", "--vid_dir", required=True, help="specify video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-g", "--gpu_ids", nargs="+", type=int, help="specify list of GPU device IDs", metavar="GPU_ID")
    args = parser.parse_args()

    predict(args.model_file, args.result_dir, args.vid_dir, args.gpu_ids)
