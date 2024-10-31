import os
import os.path as path
from glob import glob
import yaml
import script.utility as util


def split(exclude_annot_num: int, src_dirs: list[str], tgt_dir: str) -> None:
    for m in ("train", "validate", "test"):
        for t in ("images", "labels"):
            os.makedirs(path.join(tgt_dir, m, t))

    for d in src_dirs:
        pallet_cls_idx_strs = []
        with open(path.join(d, "yolo/classes.txt")) as f:
            for i, n in enumerate(f.readlines()):
                if n in ("full_palette\n", "less_palette\n", "pallet_light_load\n", "pallet_load\n"):
                    pallet_cls_idx_strs.append(str(i))

        img_files = util.random_split(glob(path.join(d, "original/*")), (0.8, 0.1, 0.1))
        for i, m in enumerate(("train", "validate", "test")):
            for img_file in img_files[i]:
                annot_file_name = path.splitext(path.basename(img_file))[0] + ".txt"
                with open(path.join(d, "yolo/annotations/", annot_file_name)) as f:
                    annot = []
                    for l in f.readlines():
                        l = l.split(" ")
                        if l[0] in pallet_cls_idx_strs:
                            annot.append("0 " + " ".join(l[1:]))
                if len(annot) > exclude_annot_num:
                    os.symlink(img_file, path.join(tgt_dir, m, "images/", path.basename(img_file)))
                    with open(path.join(tgt_dir, m, "labels/", annot_file_name), mode="w") as f:
                        f.writelines(annot)

    with open(path.join(tgt_dir, "data.yaml"), mode="w") as f:
        yaml.safe_dump({
            "path": path.abspath(tgt_dir),
            "train": "train",
            "val": "validate",
            "test": "test",
            "names": {0: "pallet"}
        }, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dirs", nargs="+", required=True, help="specify list of source dataset directories", metavar="PATH_TO_SRC_DIR")
    parser.add_argument("-t", "--tgt_dir", required=True, help="specify target dataset directory", metavar="PATH_TO_TGT_DIR")
    parser.add_argument("-n", "--exclude_annot_num", default=-1, type=int, help="maximum number of annotations to exclude images", metavar="NUM")
    args = parser.parse_args()

    split(args.exclude_annot_num, args.src_dirs, args.tgt_dir)
