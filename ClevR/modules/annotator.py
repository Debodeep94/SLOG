import re
from collections import defaultdict

def parse_caption(caption):
    # Regex to match each object in the caption
    pattern = re.compile(r'(\d+) (\w+) (\w+) (\w+) (\w+)')
    objects = pattern.findall(caption)
    return objects

def check_criteria_same_color_diff_material(caption):
    objects = parse_caption(caption)
    color_material_dict = defaultdict(set)
    
    for count, size, color, material, shape in objects:
        color_material_dict[color].add(material)
    
    for materials in color_material_dict.values():
        if len(materials) > 1:
            return 1
    return 0

def check_criteria_same_shape_diff_material(caption):
    objects = parse_caption(caption)
    shape_material_dict = defaultdict(set)
    
    for count, size, color, material, shape in objects:
        shape_material_dict[shape].add(material)
    
    for materials in shape_material_dict.values():
        if len(materials) > 1:
            return 1
    return 0

def check_criteria_same_material_same_shape_diff_color(caption):
    objects = parse_caption(caption)
    material_shape_color_dict = defaultdict(set)
    
    for count, size, color, material, shape in objects:
        material_shape_color_dict[(material, shape)].add(color)
    
    for colors in material_shape_color_dict.values():
        if len(colors) > 1:
            return 1
    return 0

def check_criteria_cubes_spheres_same_material_color(caption):
    objects = parse_caption(caption)
    material_color_dict = defaultdict(lambda: {'cube': 0, 'sphere': 0})
    
    for count, size, color, material, shape in objects:
        if shape in ['cube', 'sphere']:
            material_color_dict[(material, color)][shape] += int(count)
    
    for shapes in material_color_dict.values():
        if shapes['cube'] > 0 and shapes['sphere'] > 0:
            return 1
    return 0

def check_criteria_spheres_diff_material(caption):
    objects = parse_caption(caption)
    sphere_material_dict = defaultdict(set)
    
    for count, size, color, material, shape in objects:
        if shape == 'sphere':
            sphere_material_dict[color].add(material)
    
    for materials in sphere_material_dict.values():
        if len(materials) > 1:
            return 1
    return 0

def check_all_criteria(caption):
    return [
        check_criteria_same_color_diff_material(caption),
        check_criteria_same_shape_diff_material(caption),
        check_criteria_same_material_same_shape_diff_color(caption),
        check_criteria_cubes_spheres_same_material_color(caption),
        check_criteria_spheres_diff_material(caption)
    ]
def check_individual_criteria(caption, condition):
    if condition == 'cat1':
        return check_criteria_same_color_diff_material(caption)
    elif condition == 'cat2':
        return check_criteria_same_shape_diff_material(caption)
    elif condition == 'cat3':
        return check_criteria_same_material_same_shape_diff_color(caption)
    elif condition == 'cat4':
        return check_criteria_cubes_spheres_same_material_color(caption)
    elif condition=='cat5':
        return check_criteria_spheres_diff_material(caption)
