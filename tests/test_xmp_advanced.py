"""Behavioral coverage for XMP import, rendering, strength, and persistence."""
import json
import numpy as np
import pytest

from imaging import Recipe, apply_recipe, apply_local_preset_look
from presets import xmp_to_recipe, apply_preset_file
from xmp_support import apply_imported_grade, apply_master_curve


def xmp(tmp_path, attrs='', children='', prefix='crs'):
    path = tmp_path / 'settings.xmp'
    text = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        f'xmlns:{prefix}="http://ns.adobe.com/camera-raw-settings/1.0/">'
        f'<rdf:RDF><rdf:Description {attrs}>{children}</rdf:Description></rdf:RDF></x:xmpmeta>'
    )
    path.write_text(text, encoding='utf-8')
    return str(path)


def curve(name, points, prefix='crs'):
    return f'<{prefix}:{name}><rdf:Seq>' + ''.join(
        f'<rdf:li>{point}</rdf:li>' for point in points
    ) + f'</rdf:Seq></{prefix}:{name}>'


def ramp():
    values = np.linspace(0, 1, 128, dtype=np.float32)
    return np.broadcast_to(values[None, :, None], (24, 128, 3)).copy()


def test_namespace_independent_master_and_channel_curves(tmp_path):
    children = ''.join(curve(tag, points, 'a') for tag, points in (
        ('ToneCurvePV2012', ['0, 15', '128, 160', '255, 250']),
        ('ToneCurvePV2012Red', ['0, 0', '128, 170', '255, 255']),
        ('ToneCurvePV2012Green', ['0, 0', '255, 255']),
        ('ToneCurvePV2012Blue', ['0, 0', '128, 100', '255, 255']),
    ))
    recipe = xmp_to_recipe(xmp(tmp_path, children=children, prefix='a'))
    assert recipe.curve_mode == 'rgb'
    assert recipe.curve_points[1] == pytest.approx([128/255, 160/255])
    assert recipe.curve_r_points[1][1] == pytest.approx(170/255)
    out = apply_recipe((ramp()*255).astype(np.uint8), recipe)
    assert out[10, 64, 2] > out[10, 64, 1] > out[10, 64, 0]


def test_master_curve_is_rgb_not_lab():
    source = np.array([[[0.2, 0.4, 0.8]]], dtype=np.float32)
    actual = apply_master_curve(source, [[0, 0], [0.5, 0.25], [1, 1]])
    np.testing.assert_allclose(actual, [[[0.1, 0.2, 0.7]]], atol=1e-6)
    np.testing.assert_array_equal(apply_master_curve(source, [[0, 0], [1, 1]]), source)


@pytest.mark.parametrize('points', [['0, 0', 'nan, 10'], ['0, 0', '256, 255'], ['0, 0', '0, 10'], ['bad']])
def test_bad_curves_raise_without_mutating_base(tmp_path, points):
    base = Recipe(exposure=1.0)
    before = base.to_dict()
    with pytest.raises(ValueError):
        xmp_to_recipe(xmp(tmp_path, children=curve('ToneCurvePV2012', points)), base)
    assert base.to_dict() == before


def test_identity_curve_resets_old_curve_and_omitted_channels_survive(tmp_path):
    base = Recipe(curve_points=[[0, .2], [1, .8]], curve_b_points=[[0, .1], [1, .9]])
    result = xmp_to_recipe(xmp(tmp_path, children=curve('ToneCurvePV2012', ['0, 0', '255, 255'])), base)
    assert result.curve_points == [[0, 0], [1, 1]]
    assert result.curve_b_points == base.curve_b_points
    assert base.curve_points[0][1] == .2


def test_curves_blend_below_fifty_percent_and_respect_modules(tmp_path):
    path = xmp(tmp_path, children=curve('ToneCurvePV2012', ['0, 0', '128, 200', '255, 255']))
    r = apply_preset_file(path, strength=.25)
    assert r.curve_points[1][1] == pytest.approx((128 + .25*(200-128))/255)
    assert apply_preset_file(path, strength=0).to_dict() == Recipe().to_dict()
    assert apply_preset_file(path, modules=['Color']).curve_points == []


def test_split_and_modern_grading_fields_and_round_trip(tmp_path):
    attrs = ' '.join(f'crs:{k}="{v}"' for k,v in {
        'SplitToningShadowHue':220, 'SplitToningShadowSaturation':25,
        'SplitToningHighlightHue':40, 'SplitToningHighlightSaturation':20,
        'SplitToningBalance':15, 'ColorGradeMidtoneHue':320, 'ColorGradeMidtoneSat':18,
        'ColorGradeMidtoneLum':10, 'ColorGradeShadowLum':-8,
        'ColorGradeHighlightLum':12, 'ColorGradeGlobalHue':35,
        'ColorGradeGlobalSat':5, 'ColorGradeGlobalLum':3, 'ColorGradeBlending':70,
    }.items())
    recipe = xmp_to_recipe(xmp(tmp_path, attrs))
    assert recipe.color_grade_enabled
    assert (recipe.split_shadow_hue,recipe.split_shadow_sat) == (220,25)
    assert (recipe.grade_midtone_hue,recipe.grade_midtone_sat,recipe.grade_midtone_lum) == (320,18,10)
    assert (recipe.split_highlight_lum,recipe.split_shadow_lum) == (12,-8)
    assert (recipe.grade_global_hue,recipe.grade_global_sat,recipe.grade_global_lum) == (35,5,3)
    assert recipe.grade_blending == 70 and recipe.split_balance == 15
    path = tmp_path / 'recipe.json'
    recipe.save_json(str(path))
    loaded = Recipe.load_json(str(path))
    assert json.dumps(loaded.to_dict(), sort_keys=True) == json.dumps(recipe.to_dict(), sort_keys=True)
    np.testing.assert_array_equal(apply_recipe((ramp()*255).astype(np.uint8), loaded),
                                  apply_recipe((ramp()*255).astype(np.uint8), recipe))


def test_grading_survives_black_and_white_in_global_and_local_pipeline(tmp_path):
    path=xmp(tmp_path, 'crs:ConvertToGrayscale="True" crs:SplitToningShadowHue="35" '
             'crs:SplitToningShadowSaturation="60" crs:SplitToningHighlightHue="45" '
             'crs:SplitToningHighlightSaturation="40"')
    recipe=xmp_to_recipe(path)
    for out in (apply_recipe((ramp()*255).astype(np.uint8),recipe), apply_local_preset_look(ramp(),recipe)):
        assert out[...,2].mean() > out[...,0].mean()


def test_grade_balance_blending_and_zero_reset(tmp_path):
    base=Recipe(color_grade_enabled=True, split_shadow_hue=220,split_shadow_sat=70,
                split_highlight_hue=40,split_highlight_sat=70)
    before=apply_imported_grade(ramp(),base)
    for field,value in [('split_balance',70),('grade_blending',0),('grade_global_lum',20)]:
        changed=Recipe.from_dict(base.to_dict()); setattr(changed,field,value)
        assert np.max(np.abs(apply_imported_grade(ramp(),changed)-before)) > .01
    reset=xmp_to_recipe(xmp(tmp_path,'crs:SplitToningShadowSaturation="0" crs:SplitToningHighlightSaturation="0"'),base)
    np.testing.assert_array_equal(apply_imported_grade(ramp(),reset),ramp())


def test_grading_strength_and_module_filter(tmp_path):
    path=xmp(tmp_path,'crs:ColorGradeGlobalHue="200" crs:ColorGradeGlobalSat="40"')
    result=apply_preset_file(path,strength=.25)
    assert result.color_grade_enabled and result.grade_global_sat == 10
    assert result.grade_global_hue == 200
    assert not apply_preset_file(path,modules=['Tone']).color_grade_enabled


def test_negative_dehaze_preserved_and_reduces_contrast(tmp_path):
    path=xmp(tmp_path,'crs:Dehaze="-50"')
    recipe=xmp_to_recipe(path)
    assert recipe.clearview == -50
    assert apply_preset_file(path,strength=.5).clearview == -25
    img=np.tile(np.array([.25,.75],dtype=np.float32),64)
    img=np.broadcast_to(img[None,:,None],(32,128,3)).copy()
    assert apply_local_preset_look(img,recipe).std() < img.std()
    source=(img*255).astype(np.uint8)
    assert apply_recipe(source,recipe).std() < apply_recipe(source,Recipe()).std()


def test_auto_tone_adapts_to_image_and_explicit_values_win(tmp_path):
    path=xmp(tmp_path,'crs:AutoTone="True"')
    dark=np.full((24,24,3),30,dtype=np.uint8)
    bright=np.full((24,24,3),220,dtype=np.uint8)
    low=apply_preset_file(path,image_bgr=dark)
    high=apply_preset_file(path,image_bgr=bright)
    assert low.exposure > 0 > high.exposure
    assert apply_recipe(dark,low).mean() > dark.mean()
    assert apply_recipe(bright,high).mean() < bright.mean()
    assert abs(apply_recipe(dark,low).mean()-apply_recipe(bright,high).mean()) < 5
    with pytest.raises(ValueError, match='open image'):
        xmp_to_recipe(path)
    override=xmp_to_recipe(xmp(tmp_path,'crs:AutoTone="True" crs:Exposure2012="0.75"'),image_bgr=dark)
    assert override.exposure == .75


def test_auto_tone_bit_depth_strength_and_stable_export(tmp_path):
    path=xmp(tmp_path,'crs:AutoTone="True"')
    image=np.full((16,16,3),35,dtype=np.uint8)
    r8=apply_preset_file(path,image_bgr=image)
    r16=apply_preset_file(path,image_bgr=image.astype(np.uint16)*257)
    rf=apply_preset_file(path,image_bgr=image.astype(np.float32)/255)
    assert r8.exposure == r16.exposure == rf.exposure
    half=apply_preset_file(path,image_bgr=image,strength=.5)
    assert half.exposure == r8.exposure/2
    assert apply_preset_file(path,image_bgr=image,modules=['Color']).exposure == 0
    assert apply_preset_file(path,modules=['Color']).exposure == 0
    assert apply_preset_file(path,strength=0).exposure == 0
    saved=Recipe.from_dict(r8.to_dict())
    np.testing.assert_array_equal(apply_recipe(image,r8),apply_recipe(image,saved))
    black=np.zeros_like(image)
    assert apply_preset_file(path,image_bgr=black).exposure == 0


def test_nonfinite_scalar_is_ignored(tmp_path):
    result=xmp_to_recipe(xmp(tmp_path,'crs:Dehaze="nan" crs:ColorGradeGlobalSat="inf"'))
    assert result.clearview == 0 and not result.color_grade_enabled
