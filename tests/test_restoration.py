import json
from pathlib import Path
import cv2
import numpy as np
from restoration import (ai_confidence_map,blend_ai_result,correct_fading_silvering,
    detect_crease_scratch_mask,grain_aware_restore,load_model_pack,repair_damage,
    restore_photo,run_ai_provider,suppress_stains_texture,wiener_deblur)
from workers import write_restored_image_atomic

def test_damage_detection_and_repair_preserve_shape():
    image=np.full((120,160,3),140,np.uint8);cv2.line(image,(15,60),(145,62),(245,245,245),3);mask=detect_crease_scratch_mask(image,80,15,10);fixed=repair_damage(image,mask,4,True)
    assert mask.shape==image.shape[:2] and mask.max()==255 and fixed.shape==image.shape and fixed.dtype==image.dtype

def test_restoration_pipeline_is_bounded_and_16bit_safe():
    image=np.full((64,80,3),28000,np.uint16);image[:,20:30,2]=42000;mask=np.zeros(image.shape[:2],np.uint8);mask[20:30,30:40]=255
    result=restore_photo(image,{"stain":40,"texture":25,"fade":35,"silvering":20,"deblur_radius":1,"snr":35,"denoise":20,"sharpen":20,"repair_radius":3,"join_tears":True},mask)
    assert result.shape==image.shape and result.dtype==np.uint16 and result.min()>=0 and result.max()<=65535

def test_optional_model_pack_protocol_and_ai_blending(tmp_path):
    provider=tmp_path/"provider.py";provider.write_text("import shutil,sys\nshutil.copyfile(sys.argv[1],sys.argv[2])\n",encoding="utf-8")
    manifest={"format":"photolab-ai-model-pack-1","name":"Test Pack","providers":[{"id":"copy","name":"Copy","capabilities":["colorize","enhance"],"command":["provider.py","{input}","{output}"]}]};(tmp_path/"photolab-model-pack.json").write_text(json.dumps(manifest),encoding="utf-8")
    pack=load_model_pack(tmp_path);source=tmp_path/"source.bin";output=tmp_path/"output.bin";source.write_bytes(b"pixels");report=run_ai_provider(pack["providers"][0],source,output,"colorize",.8,1)
    assert output.read_bytes()==b"pixels" and report["returncode"]==0
    a=np.zeros((20,30,3),np.uint8);b=np.full_like(a,200);mask=np.zeros((20,30),np.uint8);mask[:,15:]=255;blended=blend_ai_result(a,b,50,mask);confidence=ai_confidence_map(a,b)
    assert blended[:,:15].max()==0 and 95<=blended[:,20:].mean()<=105 and confidence.shape==a.shape[:2]
    sixteen=np.full((10,12,3),32000,np.uint16);mixed=blend_ai_result(sixteen,np.full((10,12,3),128,np.uint8),50);assert mixed.dtype==np.uint16 and 32000<=mixed.mean()<=33000

def test_restoration_save_is_atomic_and_reports_progress(tmp_path):
    image=np.full((64,96,3),(30,90,170),np.uint8);output=tmp_path/"restored.png";events=[]
    assert write_restored_image_atomic(image,output,lambda value,message:events.append((value,message)))
    loaded=cv2.imread(str(output));assert loaded.shape==image.shape and not (tmp_path/"restored.png.photolab-part").exists()
    assert events[0][0]==8 and events[-1][0]==100 and any("Writing" in message for _,message in events)

def test_cancelled_restoration_save_leaves_no_partial_file(tmp_path):
    image=np.zeros((32,32,3),np.uint8);output=tmp_path/"cancelled.png"
    assert not write_restored_image_atomic(image,output,cancelled=lambda:True)
    assert not output.exists() and not (tmp_path/"cancelled.png.photolab-part").exists()
