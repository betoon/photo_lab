"""PhotoLab old-photo restoration and optional external AI model-pack protocol."""
from __future__ import annotations
import hashlib, json, os, shlex, subprocess, sys, tempfile
from pathlib import Path
import cv2
import numpy as np

AI_CAPABILITIES=("colorize","face_restore","reconstruct","enhance","super_resolution")

def _u8(image):
    if image.dtype==np.uint8:return image.copy(),255.0,image.dtype
    maximum=65535.0 if image.dtype==np.uint16 else max(float(np.nanmax(image)),1.0)
    return np.clip(image.astype(np.float32)/maximum*255,0,255).astype(np.uint8),maximum,image.dtype

def _restore(image,maximum,dtype):
    if np.issubdtype(dtype,np.integer):return np.clip(image.astype(np.float32)/255*maximum,0,maximum).astype(dtype)
    return np.clip(image.astype(np.float32)/255*maximum,0,maximum).astype(dtype)

def detect_crease_scratch_mask(image,sensitivity=55,min_length=18,max_width=12):
    work,_,_=_u8(image);gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY);scale=max(3,int(max_width)|1);kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(scale,scale));bright=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel);dark=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel);response=np.maximum(bright,dark);threshold=np.percentile(response,np.clip(99.5-sensitivity*.035,94,99.4));mask=(response>max(threshold,4)).astype(np.uint8)*255
    lines=cv2.HoughLinesP(cv2.Canny(gray,40,130),1,np.pi/180,max(15,min_length),minLineLength=min_length,maxLineGap=max_width*2)
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):cv2.line(mask,(int(x1),int(y1)),(int(x2),int(y2)),255,max(1,max_width//3),cv2.LINE_AA)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask);clean=np.zeros_like(mask)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>=4 and (max(w,h)>=min_length or area<=max_width*max_width*3):clean[labels==i]=255
    return cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

def repair_damage(image,mask,radius=4,join_tears=True):
    work,maximum,dtype=_u8(image);m=np.asarray(mask,np.uint8)
    if join_tears:m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    fixed=cv2.inpaint(work,m,max(1,float(radius)),cv2.INPAINT_TELEA);return _restore(fixed,maximum,dtype)

def suppress_stains_texture(image,stain_strength=0,texture_strength=0):
    work,maximum,dtype=_u8(image);out=work.astype(np.float32)
    if stain_strength:
        lab=cv2.cvtColor(work,cv2.COLOR_BGR2LAB).astype(np.float32);sigma=max(work.shape[:2])/25;local=cv2.GaussianBlur(lab,(0,0),sigma);target=np.median(local.reshape(-1,3),axis=0);lab[...,1:]+=((target-local)[...,1:])*(stain_strength/100);out=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)
    if texture_strength:
        smooth=cv2.bilateralFilter(np.clip(out,0,255).astype(np.uint8),9,35,9).astype(np.float32);out=out*(1-texture_strength/125)+smooth*(texture_strength/125)
    return _restore(np.clip(out,0,255),maximum,dtype)

def correct_fading_silvering(image,fade=0,silvering=0):
    work,maximum,dtype=_u8(image);lab=cv2.cvtColor(work,cv2.COLOR_BGR2LAB);l,a,b=cv2.split(lab);clahe=cv2.createCLAHE(1+fade/35,(8,8));enhanced=clahe.apply(l);l=cv2.addWeighted(l,1-fade/100,enhanced,fade/100,0);out=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR).astype(np.float32)
    if fade:
        med=np.median(out.reshape(-1,3),axis=0);gain=np.mean(med)/np.maximum(med,1);balanced=np.clip(out*gain,0,255);out=out*(1-fade/150)+balanced*(fade/150)
    if silvering:
        gray=cv2.cvtColor(np.clip(out,0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY);high=np.clip((gray.astype(np.float32)-170)/85,0,1)[...,None]*(silvering/100);neutral=np.repeat(gray[...,None],3,axis=2);out=out*(1-high)+neutral*high
    return _restore(np.clip(out,0,255),maximum,dtype)

def wiener_deblur(image,radius=0,snr=30):
    if radius<=0:return image.copy()
    work,maximum,dtype=_u8(image);h,w=work.shape[:2];psf=np.zeros((h,w),np.float32);cv2.circle(psf,(w//2,h//2),max(1,int(radius)),1,-1);psf/=max(psf.sum(),1);psf=np.fft.ifftshift(psf);H=np.fft.fft2(psf);filt=np.conj(H)/(np.abs(H)**2+1/max(float(snr),1));out=np.empty_like(work)
    for c in range(3):out[...,c]=np.clip(np.real(np.fft.ifft2(np.fft.fft2(work[...,c])*filt)),0,255).astype(np.uint8)
    return _restore(out,maximum,dtype)

def grain_aware_restore(image,denoise=0,sharpen=0):
    work,maximum,dtype=_u8(image);out=work
    if denoise:out=cv2.fastNlMeansDenoisingColored(work,None,denoise/5,denoise/5,7,21)
    if sharpen:
        blur=cv2.GaussianBlur(out,(0,0),1);detail=cv2.subtract(out,blur);gray=cv2.cvtColor(out,cv2.COLOR_BGR2GRAY);edges=cv2.GaussianBlur(cv2.Laplacian(gray,cv2.CV_32F),(0,0),1);weight=np.clip(np.abs(edges)/24,0,1)[...,None];out=np.clip(out.astype(np.float32)+detail.astype(np.float32)*weight*(sharpen/35),0,255).astype(np.uint8)
    return _restore(out,maximum,dtype)

def restore_photo(image,settings,mask=None):
    out=suppress_stains_texture(image,float(settings.get("stain",0)),float(settings.get("texture",0)));out=correct_fading_silvering(out,float(settings.get("fade",0)),float(settings.get("silvering",0)));out=wiener_deblur(out,float(settings.get("deblur_radius",0)),float(settings.get("snr",30)));out=grain_aware_restore(out,float(settings.get("denoise",0)),float(settings.get("sharpen",0)))
    if mask is not None and np.any(mask):out=repair_damage(out,mask,float(settings.get("repair_radius",4)),bool(settings.get("join_tears",True)))
    return out

def load_model_pack(folder):
    root=Path(folder);manifest_path=root/"photolab-model-pack.json"
    if not manifest_path.is_file():raise ValueError("photolab-model-pack.json was not found")
    data=json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("format")!="photolab-ai-model-pack-1":raise ValueError("Unsupported model-pack format")
    providers=[]
    for item in data.get("providers",[]):
        capabilities=[x for x in item.get("capabilities",[]) if x in AI_CAPABILITIES];command=item.get("command",[])
        if capabilities and isinstance(command,list) and command:providers.append({**item,"capabilities":capabilities,"root":str(root)})
    if not providers:raise ValueError("No usable providers are declared")
    return {**data,"root":str(root),"providers":providers}

def run_ai_provider(provider,input_path,output_path,capability,fidelity=.7,candidate=1,timeout=900):
    if capability not in provider.get("capabilities",[]):raise ValueError("Provider does not support this operation")
    root=Path(provider["root"]);values={"input":str(input_path),"output":str(output_path),"capability":capability,"fidelity":f"{fidelity:.3f}","candidate":str(candidate),"root":str(root)};command=[str(part).format(**values) for part in provider["command"]]
    if command[0].lower().endswith(".py"):command.insert(0,sys.executable)
    elif not os.path.isabs(command[0]):command[0]=str(root/command[0])
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=timeout,check=False)
    if result.returncode or not Path(output_path).is_file():raise RuntimeError((result.stderr or result.stdout or "Provider produced no output")[-3000:])
    return {"command":command,"returncode":result.returncode,"stdout":result.stdout[-2000:],"provider":provider.get("id",provider.get("name","provider"))}

def blend_ai_result(original,generated,strength=100,mask=None):
    base,maximum,dtype=_u8(original);gen,_gm,_gd=_u8(generated);gen=cv2.resize(gen,(base.shape[1],base.shape[0]),interpolation=cv2.INTER_LANCZOS4) if gen.shape[:2]!=base.shape[:2] else gen;alpha=np.full(base.shape[:2],np.clip(strength/100,0,1),np.float32) if mask is None else np.asarray(mask,np.float32)/255*np.clip(strength/100,0,1);out=base.astype(np.float32)*(1-alpha[...,None])+gen.astype(np.float32)*alpha[...,None];return _restore(np.clip(out,0,255),maximum,dtype)

def ai_confidence_map(original,generated):
    a,_m,_d=_u8(original);b,_m,_d=_u8(cv2.resize(generated,(a.shape[1],a.shape[0])));difference=cv2.cvtColor(cv2.absdiff(a,b),cv2.COLOR_BGR2GRAY).astype(np.float32)/255;edges=cv2.Laplacian(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.CV_32F);return np.clip((1-difference)*.75+np.exp(-np.abs(edges)/40)*.25,0,1)
