"""Generate PhotoLab's bundled, original film-rendering preset library.

The renderings describe broad photochemical aesthetics.  They do not copy LUTs,
profiles, names, or measurements from any commercial film-emulation product.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"
ZERO8 = [0.0] * 8
IDENTITY = [[0.0, 0.0], [1.0, 1.0]]


def look(category, filename, name, description, **recipe):
    base = {
        "name": name,
        "description": description,
        "preset_kind": "film_rendering",
        "film_family": category,
        "version": 1,
        "wb_as_shot": True,
        "creative_temperature": 0.0,
        "creative_tint": 0.0,
        "hsl_hue": ZERO8,
        "hsl_sat": ZERO8,
        "hsl_lum": ZERO8,
    }
    base.update(recipe)
    return category, filename, base


PRESETS = [
    # Color negative: forgiving highlights, moderate contrast, skin-aware color.
    look("Film - Color Negative", "Warm Portrait 160.json", "Warm Portrait 160",
         "Fine-grained portrait color with warm skin, gentle greens, and soft highlight roll-off.",
         contrast=-5, highlights=-18, shadows=8, blacks=3, vibrance=7, saturation=-4,
         creative_temperature=4, creative_tint=1, film_grain=7,
         hsl_hue=[0, -2, -5, -6, 0, 2, 0, 0], hsl_sat=[-3, 4, -9, -14, -8, -5, -6, -4],
         hsl_lum=[2, 7, 3, 5, 0, -2, 0, 0], curve_points=[[0,.025],[.2,.19],[.5,.51],[.82,.84],[1,.985]]),
    look("Film - Color Negative", "Natural Portrait 400.json", "Natural Portrait 400",
         "Neutral portrait rendering with open shadows, restrained reds, and natural skin separation.",
         contrast=-2, highlights=-12, shadows=7, blacks=2, clarity=-3, vibrance=4, saturation=-5,
         film_grain=13, hsl_hue=[1,-2,-3,-4,0,1,0,0], hsl_sat=[-7,2,-6,-9,-6,-4,-7,-7],
         hsl_lum=[3,6,2,2,0,-2,0,0], curve_points=[[0,.02],[.18,.18],[.5,.505],[.84,.85],[1,.99]]),
    look("Film - Color Negative", "Golden Consumer 200.json", "Golden Consumer 200",
         "Sunny consumer-film warmth with cheerful reds and yellows and a lightly lifted toe.",
         contrast=5, highlights=-7, shadows=4, vibrance=12, saturation=3, creative_temperature=8,
         creative_tint=1, film_grain=10, hsl_hue=[-2,-3,-7,-5,1,2,0,0],
         hsl_sat=[8,8,9,-5,-6,-4,0,2], hsl_lum=[2,5,5,1,0,-3,0,0],
         curve_points=[[0,.035],[.2,.21],[.5,.52],[.8,.83],[1,.99]]),
    look("Film - Color Negative", "Classic Consumer 400.json", "Classic Consumer 400",
         "Everyday color negative with lively primaries, warm highlights, and visible medium grain.",
         contrast=7, highlights=-9, shadows=3, vibrance=13, saturation=1, creative_temperature=5,
         film_grain=18, microcontrast=3, hsl_hue=[-3,-2,-4,-7,1,3,0,0],
         hsl_sat=[9,5,5,-2,-4,2,1,3], hsl_lum=[1,4,3,1,0,-4,0,0],
         curve_points=[[0,.028],[.18,.18],[.5,.51],[.82,.85],[1,.985]]),
    look("Film - Color Negative", "Muted Editorial 400.json", "Muted Editorial 400",
         "Low-saturation editorial palette with cyan-leaning greens, soft contrast, and calm skin tones.",
         contrast=-7, highlights=-15, shadows=10, blacks=4, saturation=-13, vibrance=5,
         creative_temperature=-2, creative_tint=-1, film_grain=15,
         hsl_hue=[2,0,5,14,8,-3,0,0], hsl_sat=[-9,-5,-20,-24,-13,-16,-14,-12],
         hsl_lum=[2,5,3,8,5,-2,0,0], curve_points=[[0,.045],[.22,.225],[.5,.51],[.82,.84],[1,.975]]),
    look("Film - Color Negative", "Pastel Daylight 400.json", "Pastel Daylight 400",
         "Airy pastel color with luminous skin, pale blues, lifted blacks, and protected highlights.",
         contrast=-12, highlights=-22, shadows=13, whites=-4, blacks=7, clarity=-5,
         vibrance=2, saturation=-11, creative_temperature=3, creative_tint=2, film_grain=12,
         hsl_hue=[1,-2,-4,-3,-2,2,0,0], hsl_sat=[-10,-5,-16,-18,-15,-18,-15,-12],
         hsl_lum=[5,10,8,7,7,5,3,3], curve_points=[[0,.06],[.22,.235],[.5,.525],[.8,.835],[1,.975]]),
    look("Film - Color Negative", "Cine Daylight 250.json", "Cine Daylight 250",
         "Controlled cinematic daylight color with teal shadows, warm highlights, and broad latitude.",
         contrast=2, highlights=-20, shadows=11, blacks=3, saturation=-8, vibrance=7,
         creative_temperature=2, film_grain=9, split_shadow_hue=198, split_shadow_sat=8,
         split_highlight_hue=38, split_highlight_sat=6, split_balance=8,
         hsl_hue=[0,-2,-6,8,7,-4,0,0], hsl_sat=[-4,2,-12,-17,-7,-9,-12,-10],
         hsl_lum=[1,5,1,3,1,-4,0,0], curve_points=[[0,.025],[.2,.19],[.5,.5],[.82,.85],[1,.98]]),
    look("Film - Color Negative", "Cine Tungsten 500.json", "Cine Tungsten 500",
         "Moody tungsten-stock interpretation with cool shadows, amber highlights, and pronounced grain.",
         contrast=5, highlights=-17, shadows=7, blacks=1, saturation=-9, vibrance=5,
         creative_temperature=-7, creative_tint=2, film_grain=22, split_shadow_hue=215,
         split_shadow_sat=13, split_highlight_hue=32, split_highlight_sat=9, split_balance=5,
         hsl_hue=[1,0,-4,6,5,-5,0,0], hsl_sat=[-7,0,-14,-16,-6,-4,-10,-8],
         hsl_lum=[0,3,0,2,1,-5,0,0], curve_points=[[0,.02],[.2,.185],[.5,.495],[.82,.845],[1,.975]]),
    look("Film - Color Negative", "Cool Urban 800.json", "Cool Urban 800",
         "Cool, gritty high-speed color for street and overcast scenes, with subdued yellows and greens.",
         contrast=9, highlights=-11, shadows=3, blacks=-2, clarity=4, saturation=-9,
         creative_temperature=-6, creative_tint=-1, film_grain=28, microcontrast=7,
         hsl_hue=[1,1,4,10,5,-4,0,0], hsl_sat=[-4,-5,-18,-21,-8,-5,-8,-8],
         hsl_lum=[0,2,-2,-1,0,-5,0,0], curve_points=[[0,.015],[.2,.175],[.5,.49],[.8,.82],[1,.97]]),
    look("Film - Color Negative", "Night Color 1600.json", "Night Color 1600",
         "High-speed night color with dense blacks, softened saturation, cool shade, and strong grain.",
         contrast=11, highlights=-18, shadows=5, blacks=-5, clarity=3, vibrance=4, saturation=-12,
         creative_temperature=-4, creative_tint=3, film_grain=38, microcontrast=5,
         hsl_hue=[0,1,2,7,4,-4,0,0], hsl_sat=[-7,-5,-16,-20,-9,-2,-8,-6],
         hsl_lum=[0,1,-3,-3,-1,-6,0,0], curve_points=[[0,.01],[.18,.15],[.5,.485],[.82,.84],[1,.965]]),
    look("Film - Color Negative", "Faded Family Album.json", "Faded Family Album",
         "A gently aged print look with faded blacks, warm paper highlights, and softened blues.",
         contrast=-14, highlights=-12, shadows=12, blacks=10, clarity=-7, saturation=-18,
         creative_temperature=8, creative_tint=2, film_grain=17, vignette=7,
         split_shadow_hue=190, split_shadow_sat=6, split_highlight_hue=48, split_highlight_sat=12,
         hsl_sat=[-8,-5,-14,-21,-20,-25,-18,-15], hsl_lum=[2,5,4,6,3,2,0,0],
         curve_points=[[0,.085],[.22,.245],[.5,.53],[.82,.86],[1,.96]]),
    look("Film - Color Negative", "Punchy Travel 400.json", "Punchy Travel 400",
         "Vivid travel color with deep blue skies, fresh foliage, and crisp medium-format-style contrast.",
         contrast=13, highlights=-12, shadows=5, blacks=-5, clarity=5, vibrance=18, saturation=2,
         film_grain=12, microcontrast=8, hsl_hue=[-2,-2,-5,-9,1,3,0,0],
         hsl_sat=[8,7,7,10,7,14,3,3], hsl_lum=[1,4,2,-2,-2,-7,0,0],
         curve_points=[[0,.012],[.18,.15],[.5,.5],[.82,.86],[1,.985]]),

    # Slide / cinema: stronger density and cleaner, more decisive color.
    look("Film - Slide and Cinema", "Neutral Chrome 100.json", "Neutral Chrome 100",
         "Clean, neutral transparency color with crisp midtones and restrained highlight latitude.",
         contrast=13, highlights=-8, shadows=-2, blacks=-5, clarity=3, vibrance=9, saturation=2,
         film_grain=5, hsl_sat=[2,2,0,1,1,4,0,0], hsl_lum=[0,2,0,-1,-1,-3,0,0],
         curve_points=[[0,.005],[.2,.16],[.5,.5],[.8,.85],[1,.995]]),
    look("Film - Slide and Cinema", "Vivid Landscape Chrome.json", "Vivid Landscape Chrome",
         "High-saturation landscape transparency with rich greens, deep blue skies, and firm contrast.",
         contrast=18, highlights=-10, shadows=-5, blacks=-8, clarity=5, vibrance=21, saturation=6,
         film_grain=6, microcontrast=8, hsl_hue=[-2,-3,-8,-12,1,4,0,0],
         hsl_sat=[8,9,14,22,12,24,5,4], hsl_lum=[0,2,0,-4,-4,-10,0,0],
         curve_points=[[0,0],[.18,.13],[.5,.5],[.82,.88],[1,1]]),
    look("Film - Slide and Cinema", "Warm Projector Chrome.json", "Warm Projector Chrome",
         "Warm projected-slide color with amber highlights, saturated reds, and dense cool shadows.",
         contrast=16, highlights=-7, shadows=-4, blacks=-6, vibrance=14, saturation=4,
         creative_temperature=7, creative_tint=1, film_grain=7, split_shadow_hue=218,
         split_shadow_sat=5, split_highlight_hue=42, split_highlight_sat=9, split_balance=15,
         hsl_sat=[12,9,5,-2,-3,6,2,4], hsl_lum=[-1,3,1,-2,-1,-6,0,0],
         curve_points=[[0,.005],[.2,.155],[.5,.495],[.82,.87],[1,.99]]),
    look("Film - Slide and Cinema", "Cool Mountain Chrome.json", "Cool Mountain Chrome",
         "Cool, precise transparency palette for snow, rock, water, and high-altitude blue skies.",
         contrast=15, highlights=-13, shadows=-3, blacks=-6, clarity=5, vibrance=12,
         creative_temperature=-5, film_grain=5, microcontrast=8,
         hsl_hue=[0,0,1,-4,-4,-2,0,0], hsl_sat=[0,-2,-5,5,9,14,0,0],
         hsl_lum=[0,1,0,-2,2,-8,0,0], curve_points=[[0,.005],[.2,.16],[.5,.495],[.82,.865],[1,.99]]),
    look("Film - Slide and Cinema", "Classic Reversal 64.json", "Classic Reversal 64",
         "Vintage low-speed reversal color with warm reds, muted blues, and compact dynamic range.",
         contrast=18, highlights=-5, shadows=-7, blacks=-6, saturation=-2, vibrance=10,
         creative_temperature=5, creative_tint=2, film_grain=8, hsl_hue=[-4,-3,-2,2,1,4,0,0],
         hsl_sat=[13,6,-3,-8,-9,-12,-5,-2], hsl_lum=[-2,3,0,-3,-2,-7,0,0],
         curve_points=[[0,.01],[.2,.15],[.5,.49],[.82,.87],[1,.985]]),
    look("Film - Slide and Cinema", "Bleach Bypass Cinema.json", "Bleach Bypass Cinema",
         "Silver-retention cinema aesthetic with low saturation, hard density, and crisp texture.",
         contrast=24, highlights=-18, shadows=-8, blacks=-12, clarity=9, vibrance=-8,
         saturation=-39, film_grain=24, microcontrast=16, hsl_sat=[-12,-8,-20,-22,-18,-17,-15,-12],
         curve_points=[[0,0],[.18,.105],[.5,.48],[.82,.89],[1,.99]]),
    look("Film - Slide and Cinema", "Teal Amber Cinema.json", "Teal Amber Cinema",
         "Modern cinematic separation with subtle teal shadows and amber highlights, kept skin-friendly.",
         contrast=10, highlights=-18, shadows=6, blacks=-4, saturation=-10, vibrance=7,
         film_grain=10, split_shadow_hue=198, split_shadow_sat=14, split_highlight_hue=32,
         split_highlight_sat=12, split_balance=10, hsl_hue=[0,-2,-8,10,10,-6,0,0],
         hsl_sat=[-4,5,-15,-22,-3,-11,-14,-12], hsl_lum=[0,5,0,2,1,-5,0,0],
         curve_points=[[0,.012],[.2,.17],[.5,.495],[.82,.86],[1,.98]]),
    look("Film - Slide and Cinema", "Day for Night Cinema.json", "Day for Night Cinema",
         "Stylized moonlit cinema rendering with cool density, subdued warm colors, and deep blues.",
         exposure=-0.35, contrast=15, highlights=-23, shadows=-5, blacks=-9, clarity=5,
         saturation=-22, creative_temperature=-18, creative_tint=2, film_grain=15,
         split_shadow_hue=220, split_shadow_sat=18, split_highlight_hue=205, split_highlight_sat=6,
         hsl_sat=[-18,-22,-28,-20,-2,8,-8,-10], hsl_lum=[-7,-8,-8,-3,1,-5,0,0],
         curve_points=[[0,0],[.2,.13],[.5,.44],[.82,.79],[1,.94]]),

    # B&W: HSL luminance acts as a virtual colored filter before conversion.
    look("Film - Black and White", "Fine Grain Portrait 50.json", "Fine Grain Portrait 50",
         "Smooth fine-grain monochrome with luminous skin, gentle contrast, and soft highlights.",
         black_and_white=True, contrast=-3, highlights=-17, shadows=8, blacks=2, clarity=-2,
         film_grain=6, hsl_lum=[7,13,7,1,0,-5,-2,2],
         curve_points=[[0,.02],[.2,.195],[.5,.51],[.82,.84],[1,.985]]),
    look("Film - Black and White", "Classic Panchromatic 125.json", "Classic Panchromatic 125",
         "Balanced traditional panchromatic response with crisp midtones and modest fine grain.",
         black_and_white=True, contrast=8, highlights=-10, shadows=3, blacks=-3, clarity=3,
         film_grain=10, microcontrast=4, hsl_lum=[4,7,5,1,-2,-5,-2,1],
         curve_points=[[0,.008],[.2,.17],[.5,.5],[.82,.86],[1,.99]]),
    look("Film - Black and White", "Documentary 400.json", "Documentary 400",
         "Classic documentary tonality with strong local contrast and an honest medium grain.",
         black_and_white=True, contrast=13, highlights=-13, shadows=3, blacks=-6, clarity=7,
         film_grain=19, microcontrast=10, hsl_lum=[3,5,2,0,-2,-4,-2,0],
         curve_points=[[0,.005],[.18,.135],[.5,.49],[.82,.87],[1,.99]]),
    look("Film - Black and White", "Street 800.json", "Street 800",
         "Gritty high-speed street monochrome with dense blacks, punchy midtones, and coarse grain.",
         black_and_white=True, contrast=19, highlights=-15, shadows=-2, blacks=-10, clarity=10,
         film_grain=31, microcontrast=15, hsl_lum=[2,3,0,-2,-3,-5,-2,0], vignette=5,
         curve_points=[[0,0],[.18,.105],[.5,.475],[.82,.885],[1,.985]]),
    look("Film - Black and White", "Available Light 1600.json", "Available Light 1600",
         "Open-shadow high-speed monochrome for dim interiors, with soft highlights and pronounced grain.",
         black_and_white=True, contrast=7, highlights=-24, shadows=14, blacks=-3, clarity=5,
         film_grain=40, microcontrast=8, hsl_lum=[5,7,3,0,-2,-5,-2,1],
         curve_points=[[0,.012],[.18,.155],[.5,.49],[.82,.845],[1,.97]]),
    look("Film - Black and White", "Yellow Filter Landscape.json", "Yellow Filter Landscape",
         "Natural landscape monochrome with gently bright foliage and moderately darkened blue skies.",
         black_and_white=True, contrast=12, highlights=-12, shadows=3, blacks=-5, clarity=7,
         film_grain=12, microcontrast=10, hsl_lum=[6,11,14,9,0,-18,-8,0],
         curve_points=[[0,.005],[.2,.16],[.5,.5],[.82,.87],[1,.99]]),
    look("Film - Black and White", "Red Filter Dramatic.json", "Red Filter Dramatic",
         "Dramatic red-filter rendering with pale skin and clouds, dark skies, and forceful contrast.",
         black_and_white=True, contrast=21, highlights=-17, shadows=-3, blacks=-10, clarity=9,
         film_grain=14, microcontrast=14, hsl_lum=[25,20,5,-12,-22,-38,-25,-8],
         curve_points=[[0,0],[.18,.105],[.5,.48],[.82,.89],[1,.99]]),
    look("Film - Black and White", "Green Filter Portrait.json", "Green Filter Portrait",
         "Green-filter portrait and botanical rendering with separated foliage and controlled reds.",
         black_and_white=True, contrast=7, highlights=-15, shadows=6, blacks=-2, clarity=2,
         film_grain=11, hsl_lum=[-6,4,11,22,9,-10,-5,-3],
         curve_points=[[0,.012],[.2,.18],[.5,.505],[.82,.855],[1,.985]]),
    look("Film - Black and White", "Orthochromatic Vintage.json", "Orthochromatic Vintage",
         "Early-film-inspired response with dark reds, pale blues, lifted blacks, and visible grain.",
         black_and_white=True, contrast=10, highlights=-8, shadows=5, blacks=5, clarity=1,
         film_grain=22, vignette=7, hsl_lum=[-30,-16,4,12,16,22,12,-18],
         curve_points=[[0,.055],[.2,.19],[.5,.49],[.82,.86],[1,.975]]),
    look("Film - Black and White", "Infrared Dream 720.json", "Infrared Dream 720",
         "Creative infrared-like monochrome with luminous foliage, dark skies, glow, and coarse grain.",
         black_and_white=True, ir_mono=True, contrast=19, highlights=-22, shadows=5, blacks=-9,
         clarity=-5, film_grain=24, microcontrast=6, vignette=8,
         hsl_lum=[18,20,26,38,2,-45,-30,-10], curve_points=[[0,0],[.18,.12],[.5,.51],[.82,.9],[1,.985]]),
]


def main():
    expected = {category for category, _, _ in PRESETS}
    for category in expected:
        folder = PLUGIN / category
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.json"):
            old.unlink()
    for category, filename, data in PRESETS:
        path = PLUGIN / category / filename
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(PRESETS)} film presets in {len(expected)} categories.")


if __name__ == "__main__":
    main()
