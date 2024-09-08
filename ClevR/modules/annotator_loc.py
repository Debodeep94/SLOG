import re
from collections import defaultdict

# Function to parse the caption
def parse_caption(caption):
    pattern = re.compile(r'(\d+) (\w+) (\w+) (\w+) (\w+)')
    objects = pattern.findall(caption)
    return objects

# General function to check if an object is behind another in the list
def is_behind(object1, object2, objects):
    indices = {tuple(obj): idx for idx, obj in enumerate(objects)}
    return indices.get(object1, float('inf')) > indices.get(object2, float('inf'))

# Function to match specific properties in parsed objects
def find_objects_with_properties(objects, color=None, shape=None, material=None):
    return [
        obj for obj in objects 
        if (color is None or obj[2] == color) and 
           (shape is None or obj[4] == shape) and
           (material is None or obj[3] == material)
    ]

# # Rule 1: cylinder behind cylinder
def rule_red_behind_grey(caption):
    objects = parse_caption(caption)
    cylinders = find_objects_with_properties(objects, shape='cylinder')
    cubes = find_objects_with_properties(objects, shape='cylinder')
    
    for cylinder in cylinders:
        for cube in cubes:
            if is_behind(cylinder, cube, objects):
                return 1
    return 0

# Rule 2: Cylinder behind cube
def rule_cylinder_behind_cube(caption):
    objects = parse_caption(caption)
    cylinders = find_objects_with_properties(objects, shape='cylinder')
    cubes = find_objects_with_properties(objects, shape='cube')
    
    for cylinder in cylinders:
        for cube in cubes:
            if is_behind(cylinder, cube, objects):
                return 1
    return 0

# Rule 3: Cube behind brown object
def rule_cube_behind_brown(caption):
    objects = parse_caption(caption)
    cubes = find_objects_with_properties(objects, shape='cube')
    brown_objects = find_objects_with_properties(objects, color='brown')
    
    for cube in cubes:
        for brown in brown_objects:
            if is_behind(cube, brown, objects):
                return 1
    return 0

# Rule 4: Spheres with different material
def rule_spheres_diff_material(caption):
    objects = parse_caption(caption)
    spheres = defaultdict(set)
    
    for obj in objects:
        if obj[4] == 'sphere':
            spheres[obj[2]].add(obj[3])  # color -> material set
    
    for material_set in spheres.values():
        if len(material_set) > 1:
            return 1
    return 0

# Rule 5: Sphere behind sphere
def rule_sphere_behind_sphere(caption):
    objects = parse_caption(caption)
    spheres = find_objects_with_properties(objects, shape='sphere')
    
    for i in range(len(spheres) - 1):
        if is_behind(spheres[i + 1], spheres[i], objects):
            return 1
    return 0

# Rule 6: Cube behind brown cube
def rule_cube_behind_brown_cube(caption):
    objects = parse_caption(caption)
    cubes = find_objects_with_properties(objects, shape='cube')
    brown_cubes = find_objects_with_properties(objects, shape='cube', color='brown')
    
    for cube in cubes:
        for brown_cube in brown_cubes:
            if is_behind(cube, brown_cube, objects):
                return 1
    return 0

# Rule 7: Red cylinder behind blue sphere
def rule_red_cylinder_behind_blue_sphere(caption):
    objects = parse_caption(caption)
    red_cylinders = find_objects_with_properties(objects, shape='cylinder', color='red')
    blue_spheres = find_objects_with_properties(objects, shape='sphere', color='blue')
    
    for red_cylinder in red_cylinders:
        for blue_sphere in blue_spheres:
            if is_behind(red_cylinder, blue_sphere, objects):
                return 1
    return 0

# Rule 8: Cylinder behind cylinder
def rule_cylinder_behind_cylinder(caption):
    objects = parse_caption(caption)
    cylinders = find_objects_with_properties(objects, shape='cylinder')
    
    for i in range(len(cylinders) - 1):
        if is_behind(cylinders[i + 1], cylinders[i], objects):
            return 1
    return 0

# Rule 9: Cube behind cube
def rule_cube_behind_cube(caption):
    objects = parse_caption(caption)
    cubes = find_objects_with_properties(objects, shape='cube')
    
    for i in range(len(cubes) - 1):
        if is_behind(cubes[i + 1], cubes[i], objects):
            return 1
    return 0

# Rule 10: Any color behind the same color object
def rule_same_color_behind(caption):
    objects = parse_caption(caption)
    color_dict = defaultdict(list)
    
    for obj in objects:
        color_dict[obj[2]].append(obj)
    
    for color_objects in color_dict.values():
        for i in range(len(color_objects) - 1):
            if is_behind(color_objects[i + 1], color_objects[i], objects):
                return 1
    return 0


def check_all_criteria(caption):
    return [
            # rule_red_behind_grey(caption),
            rule_cylinder_behind_cube(caption),
            rule_cube_behind_brown(caption),
            rule_spheres_diff_material(caption),
            # rule_sphere_behind_sphere(caption),
            rule_cube_behind_brown_cube(caption),
            rule_red_cylinder_behind_blue_sphere(caption)
            # rule_cylinder_behind_cylinder(caption),
            # rule_cube_behind_cube(caption),
            # rule_same_color_behind(caption)
    ]