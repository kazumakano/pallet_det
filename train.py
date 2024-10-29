import os.path as path
import pickle
from typing import Optional
from torch import cuda
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
import script.utility as util


def train(data_file: str, model_file: str, only_test: bool, param_file: str, result_dir_name: str, gpu_ids: Optional[list[int]] = None) -> None:
    if gpu_ids is None:
        gpu_ids = list(range(cuda.device_count()))
    param = util.load_param(param_file)

    if not only_test:
        YOLO(model=model_file).train(
            data=data_file,
            epochs=param["epoch"],
            batch=param["batch_size"],
            device=gpu_ids,
            workers=param["num_workers"],
            project="result",
            name=result_dir_name,
            translate=param["max_shift"],
            scale=param["max_scale"],
            shear=param["max_shear_angle"],
            perspective=param["max_perspective"],
            flipud=0.5,
            mosaic=0
        )
        model_file = path.join(path.dirname(__file__), "result/", result_dir_name, "weights/best.pt")

    result: DetMetrics = YOLO(model=model_file).val(
        data=data_file,
        batch=param["batch_size"],
        device=gpu_ids,
        split="test",
        project="result",
        name=result_dir_name
    )
    with open(path.join(result.save_dir, "test_results.pkl"), mode="wb") as f:
        pickle.dump(result.box, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", required=True, help="specify data file", metavar="PATH_TO_DATA_FILE")
    parser.add_argument("-m", "--model_file", required=True, help="specify pretrained model file", metavar="PATH_TO_MODEL_FILE")
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-r", "--result_dir_name", required=True, help="specify result directory name", metavar="RESULT_DIR_NAME")
    parser.add_argument("-g", "--gpu_ids", nargs="*", type=int, help="specify list of GPU device IDs", metavar="GPU_ID")
    parser.add_argument("-t", "--only_test", action="store_true", help="run only test")
    args = parser.parse_args()

    train(args.data_file, args.model_file, args.only_test, args.param_file, args.result_dir_name, args.gpu_ids)
