import argparse
from pathlib import Path
import cv2

TASK_ROOT=Path("/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplingnone/task2")
DIR_MAP={"rgb":"rgb","gradcam":"feature_gradcam"}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--task-root",type=Path,default=TASK_ROOT)
    p.add_argument("--fully-root",type=Path,default=None)
    p.add_argument("--mt-root",type=Path,default=None)
    p.add_argument("--dg-root",type=Path,default=None)
    p.add_argument("--out-root",type=Path,default=None)
    return p.parse_args()

def get_ratio_map(root):
    ratio_map={}
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or "_labeled_" not in exp_dir.name:continue
        ratio=exp_dir.name.split("_labeled_",1)[0]
        fold_dir=exp_dir/"f0"
        if fold_dir.is_dir():ratio_map[ratio]=fold_dir
    return ratio_map

def get_common_names(*dirs):
    names=None
    for d in dirs:
        cur={p.name for p in d.iterdir() if p.is_file()}
        names=cur if names is None else names&cur
    return sorted(names)

def read_rgb(path):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is None:raise FileNotFoundError(path)
    return img

def crop_right_half(img):
    h,w=img.shape[:2]
    return img[:,w//2:,:]

def concat_images(paths,out_path,mode):
    imgs=[read_rgb(path) for path in paths]
    if mode=="rgb":imgs=[imgs[0],crop_right_half(imgs[1]),crop_right_half(imgs[2])]
    h=imgs[0].shape[0]
    imgs=[img if img.shape[0]==h else cv2.resize(img,(int(round(img.shape[1]*h/img.shape[0])),h),interpolation=cv2.INTER_LINEAR) for img in imgs]
    out_path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out_path),cv2.hconcat(imgs))

def main():
    args=parse_args()
    fully_root=args.fully_root or args.task_root/"课程_Fully"
    mt_root=args.mt_root or args.task_root/"课程_MT"
    dg_root=args.dg_root or args.task_root/"课程_MT_depth_guider_v3"
    out_root=args.out_root or args.task_root/"concat"
    fully_map=get_ratio_map(fully_root)
    mt_map=get_ratio_map(mt_root)
    dg_map=get_ratio_map(dg_root)
    for ratio in sorted(set(fully_map)&set(mt_map)&set(dg_map),key=lambda x:(len(x),x)):
        for out_name,subdir in DIR_MAP.items():
            fully_dir=fully_map[ratio]/subdir
            mt_dir=mt_map[ratio]/subdir
            dg_dir=dg_map[ratio]/subdir
            if not fully_dir.is_dir() or not mt_dir.is_dir() or not dg_dir.is_dir():continue
            for name in get_common_names(fully_dir,mt_dir,dg_dir):
                concat_images([fully_dir/name,mt_dir/name,dg_dir/name],out_root/ratio/out_name/name,out_name)

if __name__=="__main__":
    main()
