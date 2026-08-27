import cv2
import numpy as np

from distractions import (apply_distraction_operations, apply_reflection_adjustment,
    build_sensor_dust_map, detect_dust_spots, operations_mask, reflection_mask,
    edit_reflection_mask, separate_reflections, smart_object_mask)
from imaging import Recipe, apply_recipe


def test_heal_and_clone_preserve_shape_and_16_bit_depth():
    image=np.full((80,100,3),30000,np.uint16); image[38:43,48:53]=0
    operations=[{"type":"heal","x":.5,"y":.5,"radius":.08},
                {"type":"clone","source_x":.2,"source_y":.2,"x":.75,"y":.7,"radius":.05}]
    result=apply_distraction_operations(image,operations)
    assert result.shape==image.shape and result.dtype==np.uint16
    assert result[40,50].mean()>image[40,50].mean()


def test_operation_mask_contains_circle_and_wire():
    ops=[{"type":"heal","x":.5,"y":.5,"radius":.1},
         {"type":"wire","points":[(.1,.1),(.9,.9)],"radius":.01}]
    mask=operations_mask((100,100,3),ops)
    assert mask.dtype==np.uint8 and mask[50,50]>0 and mask[10,10]>0


def test_dust_detection_and_reusable_map_find_fixed_spot():
    images=[]
    for value in (115,125,135):
        image=np.full((160,180,3),value,np.uint8)
        cv2.circle(image,(70,60),4,(10,10,10),-1); images.append(image)
    assert detect_dust_spots(images[0],85,15)
    recurring=build_sensor_dust_map(images,85,.5,15)
    assert recurring and any(abs(x["x"]-70/179)<.05 for x in recurring)


def test_reflection_mask_and_adjustment_are_bounded():
    image=np.full((100,120,3),80,np.uint8); image[30:70,40:80]=250
    mask=reflection_mask(image,70,2)
    result=apply_reflection_adjustment(image,mask,80,-60,0,20,10)
    assert mask[50,60]>mask[5,5]
    assert result.dtype==np.uint8 and result[50,60].mean()<image[50,60].mean()


def test_reflection_mask_paint_and_erase_strokes():
    mask=np.zeros((100,120),np.uint8)
    painted=edit_reflection_mask(mask,[{"x":.5,"y":.5,"radius":.12,"mode":"add"}])
    erased=edit_reflection_mask(painted,[{"x":.5,"y":.5,"radius":.06,"mode":"erase"}])
    assert painted[50,60]>0
    assert erased[50,60]<painted[50,60]


def test_multiframe_reflection_separation_outputs_diagnostics():
    base=np.full((80,100,3),60,np.uint8)
    a=base.copy(); b=base.copy(); cv2.circle(a,(30,40),10,(230,230,230),-1); cv2.circle(b,(65,40),10,(230,230,230),-1)
    clean,layer,mask,confidence,diagnostics=separate_reflections([a,b])
    assert clean.shape==a.shape and layer.shape==a.shape
    assert mask.shape==a.shape[:2] and confidence.shape==a.shape[:2]
    assert len(diagnostics)==2


def test_smart_mask_and_recipe_round_trip():
    image=np.zeros((80,100,3),np.uint8); image[20:60,30:70]=(200,180,160)
    mask=smart_object_mask(image,(.2,.1,.8,.9))
    assert mask.shape==image.shape[:2] and mask.max()==255
    recipe=Recipe(distraction_operations=[{"type":"heal","x":.5,"y":.5,"radius":.05}],reflection_enabled=True,
                  reflection_mask_strokes=[{"x":.4,"y":.4,"radius":.05,"mode":"add"}])
    restored=Recipe.from_dict(recipe.to_dict())
    result=apply_recipe(image,restored)
    assert restored.reflection_enabled and restored.reflection_mask_strokes and result.shape==image.shape
