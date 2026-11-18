"""COCO instance polygons/RLE -> aligned fractional 16x16 occupancy targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def decode_rle(seg, height, width):
    size = tuple(map(int, seg["size"]))
    if size != (height, width): raise ValueError(f"RLE shape {size} != {(height,width)}")
    counts = seg["counts"]
    if isinstance(counts, (str, bytes)):
        s = counts.decode("ascii") if isinstance(counts, bytes) else counts
        runs=[]; pos=0
        while pos<len(s):
            x=0; shift=0
            while True:
                c=ord(s[pos])-48; pos+=1; x|=(c&0x1f)<<shift; more=c&0x20; shift+=5
                if not more:
                    if c&0x10: x |= -1 << shift
                    break
            if len(runs)>2: x += runs[-2]
            runs.append(x)
        counts=runs
    flat=np.zeros(height*width,dtype=np.uint8); offset=0; val=0
    for run in counts:
        end=offset+int(run)
        if int(run)<0 or end>flat.size: raise ValueError("Invalid COCO RLE")
        if val: flat[offset:end]=1
        offset=end; val=1-val
    if offset != flat.size: raise ValueError("COCO RLE length mismatch")
    return flat.reshape((width,height)).T.copy()


def decode_segmentation(seg, height, width):
    if isinstance(seg, dict): return decode_rle(seg,height,width)
    canvas=Image.new("L",(width,height),0); draw=ImageDraw.Draw(canvas)
    for poly in seg or []:
        if len(poly)>=6: draw.polygon(list(zip(poly[0::2],poly[1::2])),fill=1)
    return np.asarray(canvas,dtype=np.uint8)


def geometry_from_transform(transform):
    """Extract exact resize/crop parameters from the OpenVLA policy processor."""
    if hasattr(transform,"tvf_resize_params") and hasattr(transform,"tvf_crop_params"):
        resize=transform.tvf_resize_params[0];crop=transform.tvf_crop_params[0]
        return {"resize_size":resize.get("size"),"resize_interpolation":resize.get("interpolation"),
                "crop_size":tuple(map(int,crop["output_size"])),"letterbox":bool(getattr(transform,"tvf_do_letterbox",False)),
                "letterbox_fill":getattr(transform,"tvf_letterbox_fill",None),
                "resize_strategy":getattr(transform,"image_resize_strategy",None),
                "processor_type":type(transform).__name__}
    """Extract Resize/CenterCrop geometry from a torchvision transform."""
    transforms = getattr(transform, "transforms", None)
    if transforms is None: raise TypeError(f"Unsupported native DINO transform {transform!r}")
    resize = next((x for x in transforms if x.__class__.__name__ == "Resize"), None)
    crop = next((x for x in transforms if x.__class__.__name__ == "CenterCrop"), None)
    if crop is None: raise ValueError("OpenVLA DINO preprocessing has no CenterCrop; implement geometry explicitly")
    size = crop.size
    if isinstance(size, int): crop_size=(size,size)
    else: crop_size=tuple(map(int,size))
    return {"resize_size": resize.size if resize is not None else None,
            "resize_interpolation": str(getattr(resize,"interpolation",None)),
            "crop_size": crop_size,
            "transform": repr(transform)}


def _resize_short_edge(h,w,out):
    if w<h: return int(out*h/w),out
    return out,int(out*w/h)


def apply_mask_geometry(mask, geometry):
    arr=np.asarray(mask,dtype=np.uint8); h,w=arr.shape; rh,rw=h,w
    if geometry.get("letterbox"):
        from PIL import ImageOps
        wfill,hfill=0,0
        side=max(h,w);top=(side-h)//2;left=(side-w)//2
        pad=Image.new("L",(side,side),0);pad.paste(Image.fromarray(arr),(left,top));arr=np.asarray(pad,dtype=np.uint8);h,w=arr.shape
    size=geometry["resize_size"]
    if size is not None:
        if isinstance(size,int): rh,rw=_resize_short_edge(h,w,size)
        elif len(size)==2: rh,rw=map(int,size)
        else: raise ValueError(f"Unexpected resize size {size}")
        arr=np.asarray(Image.fromarray(arr).resize((rw,rh),Image.Resampling.NEAREST),dtype=np.uint8)
    ch,cw=geometry["crop_size"]; top=round((rh-ch)/2); left=round((rw-cw)/2)
    # Match torchvision CenterCrop's zero padding when input is smaller than crop.
    canvas=np.zeros((max(rh,ch),max(rw,cw)),dtype=np.uint8)
    dy=(canvas.shape[0]-rh)//2; dx=(canvas.shape[1]-rw)//2; canvas[dy:dy+rh,dx:dx+rw]=arr
    top=round((canvas.shape[0]-ch)/2); left=round((canvas.shape[1]-cw)/2)
    return canvas[top:top+ch,left:left+cw]


def fractional_grid(mask224, grid=16):
    h,w=mask224.shape
    if h%grid or w%grid: raise ValueError("224 geometry must partition evenly into 16x16 cells")
    kh,kw=h//grid,w//grid
    return mask224.astype(np.float32).reshape(grid,kh,grid,kw).mean((1,3)).reshape(-1)


def extract_instances_cache(annotation_json: Path, image_ids, image_info, geometry, output_dir: Path,
                            grid=16, progress=None, split="train2014"):
    """Build resumable float16 occupancy/valid caches, indexed by supplied image_ids."""
    payload=json.loads(Path(annotation_json).read_text())
    cats=sorted(payload["categories"],key=lambda c:int(c["id"]))
    cat_ids=[int(c["id"]) for c in cats]; cat_names=[c["name"] for c in cats]; cat_row={x:i for i,x in enumerate(cat_ids)}
    by_image={int(i):[] for i in image_ids}
    for ann in payload["annotations"]:
        iid=int(ann["image_id"])
        if iid in by_image: by_image[iid].append(ann)
    n=len(image_ids); c=len(cat_ids); p=grid*grid
    output_dir.mkdir(parents=True,exist_ok=True)
    occ_path=output_dir/f"occupancy_{split}_fp16.npy"; valid_path=output_dir/f"valid_{split}_fp16.npy"
    done_path=output_dir/f"mask_done_{split}.npy"
    is_new=not done_path.exists()
    occ=np.lib.format.open_memmap(occ_path,mode="r+" if occ_path.exists() else "w+",dtype=np.float16,shape=(n,c,p))
    valid=np.lib.format.open_memmap(valid_path,mode="r+" if valid_path.exists() else "w+",dtype=np.float16,shape=(n,p))
    done=np.lib.format.open_memmap(done_path,mode="r+" if not is_new else "w+",dtype=np.uint8,shape=(n,))
    if is_new: done[:]=0; occ[:]=0; valid[:]=1; done.flush(); occ.flush(); valid.flush()
    for row,iid0 in enumerate(image_ids):
        if done[row]: continue
        iid=int(iid0); info=image_info[iid]; h,w=int(info["height"]),int(info["width"])
        unions={}; ignored=np.zeros((h,w),np.uint8)
        for ann in by_image.get(iid,[]):
            mask=decode_segmentation(ann.get("segmentation",[]),h,w)
            if ann.get("iscrowd",0): ignored |= mask
            else:
                ci=cat_row.get(int(ann["category_id"]))
                if ci is not None:
                    if ci not in unions: unions[ci]=np.zeros((h,w),np.uint8)
                    unions[ci] |= mask
        valid[row]=fractional_grid(apply_mask_geometry(1-ignored,geometry),grid).astype(np.float16)
        occ[row]=0
        for ci,m in unions.items(): occ[row,ci]=fractional_grid(apply_mask_geometry(m,geometry),grid).astype(np.float16)
        # Flush values before durable completion flag.
        if row%32==31 or row==n-1: occ.flush(); valid.flush()
        done[row]=1
        if row%32==31 or row==n-1:
            done.flush()
            if progress: progress(row+1,n)
    occ.flush(); valid.flush(); done.flush()
    return {"occupancy":str(occ_path),"valid":str(valid_path),"done":str(done_path),
            "shape":[n,c,p],"categories":[{"id":cat_ids[i],"name":cat_names[i]} for i in range(c)]}
