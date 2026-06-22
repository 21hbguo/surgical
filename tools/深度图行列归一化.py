import os,cv2,multiprocessing as mp,numpy as np,argparse
from scipy.ndimage import minimum_filter
def global_norm(x):
    dn=x-np.min(x,axis=1,keepdims=True)
    dn=dn-np.min(dn,axis=0,keepdims=True)
    return dn
def local_norm(x,win=65):
    local_min=minimum_filter(x,size=win)
    return x-local_min
def run(path,inp,out):
    img=cv2.imread(path)
    h,w=img.shape[:2]
    img=img[20:h-20,20:w-20]
    if len(img.shape)==2:
        g=global_norm(img.astype(np.float32))
        l=local_norm(img.astype(np.float32))
    else:
        g=np.stack([global_norm(img[...,c].astype(np.float32))for c in range(3)],axis=-1)
        l=np.stack([local_norm(img[...,c].astype(np.float32))for c in range(3)],axis=-1)
    g=(g-np.min(g))/(np.max(g)-np.min(g)+1e-6)*255
    l=(l-np.min(l))/(np.max(l)-np.min(l)+1e-6)*255
    comb=np.hstack([img,g.astype(np.uint8),l.astype(np.uint8)])
    cv2.imwrite(os.path.join(out,os.path.basename(path)),comb)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=str,required=True)
    ap.add_argument("--output",type=str,required=True)
    args=ap.parse_args()
    os.makedirs(args.output,exist_ok=True)
    fl=[os.path.join(args.input,f)for f in os.listdir(args.input)if f.endswith(".png")]
    pool=mp.Pool(mp.cpu_count())
    pool.starmap(run,[(f,args.input,args.output)for f in fl])
    pool.close();pool.join()