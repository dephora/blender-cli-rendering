import sys
import math
from typing import Any, Dict, Iterable, List, Tuple, Optional
import bpy
import mathutils
import numpy as np


# ============================================================================
# ARMATURE
# =============================================================================

def create_armature_mesh(scene: bpy.types.Scene, armature_object: bpy.types.Object, mesh_name: str) -> bpy.types.Object:
    assert armature_object.type == 'ARMATURE', 'Error'
    assert len(armature_object.data.bones) != 0, 'Error'

    def add_rigid_vertex_group(target_object: bpy.types.Object, name: str, vertex_indices: Iterable[int]) -> None:
        new_vertex_group = target_object.vertex_groups.new(name=name)
        for vertex_index in vertex_indices:
            new_vertex_group.add([vertex_index], 1.0, 'REPLACE')

    def generate_bone_mesh_pydata(radius: float, length: float) -> Tuple[List[mathutils.Vector], List[List[int]]]:
        base_radius = radius
        top_radius = 0.5 * radius

        vertices = [
            # Cross-section of the base part
            mathutils.Vector((-base_radius, 0.0, +base_radius)),
            mathutils.Vector((+base_radius, 0.0, +base_radius)),
            mathutils.Vector((+base_radius, 0.0, -base_radius)),
            mathutils.Vector((-base_radius, 0.0, -base_radius)),

            # Cross-section of the top part
            mathutils.Vector((-top_radius, length, +top_radius)),
            mathutils.Vector((+top_radius, length, +top_radius)),
            mathutils.Vector((+top_radius, length, -top_radius)),
            mathutils.Vector((-top_radius, length, -top_radius)),

            # End points
            mathutils.Vector((0.0, -base_radius, 0.0)),
            mathutils.Vector((0.0, length + top_radius, 0.0))
        ]

        faces = [
            # End point for the base part
            [8, 1, 0],
            [8, 2, 1],
            [8, 3, 2],
            [8, 0, 3],

            # End point for the top part
            [9, 4, 5],
            [9, 5, 6],
            [9, 6, 7],
            [9, 7, 4],

            # Side faces
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]

        return vertices, faces

    armature_data: bpy.types.Armature = armature_object.data

    vertices: List[mathutils.Vector] = []
    faces: List[List[int]] = []
    vertex_groups: List[Dict[str, Any]] = []

    for bone in armature_data.bones:
        radius = 0.10 * (0.10 + bone.length)
        temp_vertices, temp_faces = generate_bone_mesh_pydata(radius, bone.length)

        vertex_index_offset = len(vertices)

        temp_vertex_group = {'name': bone.name, 'vertex_indices': []}
        for local_index, vertex in enumerate(temp_vertices):
            vertices.append(bone.matrix_local @ vertex)
            temp_vertex_group['vertex_indices'].append(local_index + vertex_index_offset)
        vertex_groups.append(temp_vertex_group)

        for face in temp_faces:
            if len(face) == 3:
                faces.append([
                    face[0] + vertex_index_offset,
                    face[1] + vertex_index_offset,
                    face[2] + vertex_index_offset,
                ])
            else:
                faces.append([
                    face[0] + vertex_index_offset,
                    face[1] + vertex_index_offset,
                    face[2] + vertex_index_offset,
                    face[3] + vertex_index_offset,
                ])

    new_object = create_mesh_from_pydata(scene, vertices, faces, mesh_name, mesh_name)
    new_object.matrix_world = armature_object.matrix_world

    for vertex_group in vertex_groups:
        add_rigid_vertex_group(new_object, vertex_group['name'], vertex_group['vertex_indices'])

    armature_modifier = new_object.modifiers.new('Armature', 'ARMATURE')
    armature_modifier.object = armature_object
    armature_modifier.use_vertex_groups = True

    add_subdivision_surface_modifier(new_object, 1, is_simple=True)
    add_subdivision_surface_modifier(new_object, 2, is_simple=False)

    # Set the armature as the parent of the new object
    bpy.ops.object.select_all(action='DESELECT')
    new_object.select_set(True)
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.parent_set(type='OBJECT')

    return new_object


# ============================================================================
# ARMATURE END
# =============================================================================

# =============================================================================
# CAMERA
# =============================================================================

def create_camera(location: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=location)

    return bpy.context.object


def set_camera_params(camera: bpy.types.Camera,
                      focus_target_object: bpy.types.Object,
                      lens: float = 85.0,
                      fstop: float = 1.4) -> None:
    # Simulate Sony's FE 85mm F1.4 GM
    camera.sensor_fit = 'HORIZONTAL'
    camera.sensor_width = 36.0
    camera.sensor_height = 24.0
    camera.lens = lens
    camera.dof.use_dof = True
    camera.dof.focus_object = focus_target_object
    camera.dof.aperture_fstop = fstop
    camera.dof.aperture_blades = 11


# =============================================================================
# CAMERA END
# =============================================================================


# =============================================================================
# COMPOSITION
# =============================================================================


def add_split_tone_node_group() -> bpy.types.NodeGroup:
    group = bpy.data.node_groups.new(type="CompositorNodeTree", name="SplitToneSub")

    input_node = group.nodes.new("NodeGroupInput")
    group.inputs.new("NodeSocketColor", "Image")
    group.inputs.new("NodeSocketFloat", "Hue")
    group.inputs.new("NodeSocketFloat", "Saturation")

    solid_node = group.nodes.new(type="CompositorNodeCombHSVA")
    solid_node.inputs["S"].default_value = 1.0
    solid_node.inputs["V"].default_value = 1.0
    solid_node.inputs["A"].default_value = 1.0

    input_sep_node = group.nodes.new(type="CompositorNodeSepHSVA")

    overlay_node = group.nodes.new(type="CompositorNodeMixRGB")
    overlay_node.blend_type = 'OVERLAY'

    overlay_sep_node = group.nodes.new(type="CompositorNodeSepHSVA")

    comb_node = group.nodes.new(type="CompositorNodeCombHSVA")

    output_node = group.nodes.new("NodeGroupOutput")
    group.outputs.new("NodeSocketColor", "Image")

    group.links.new(input_node.outputs["Hue"], solid_node.inputs["H"])
    group.links.new(input_node.outputs["Saturation"], overlay_node.inputs["Fac"])
    group.links.new(input_node.outputs["Image"], overlay_node.inputs[1])
    group.links.new(solid_node.outputs["Image"], overlay_node.inputs[2])
    group.links.new(overlay_node.outputs["Image"], overlay_sep_node.inputs["Image"])
    group.links.new(input_node.outputs["Image"], input_sep_node.inputs["Image"])
    group.links.new(overlay_sep_node.outputs["H"], comb_node.inputs["H"])
    group.links.new(overlay_sep_node.outputs["S"], comb_node.inputs["S"])
    group.links.new(input_sep_node.outputs["V"], comb_node.inputs["V"])
    group.links.new(input_sep_node.outputs["A"], comb_node.inputs["A"])
    group.links.new(comb_node.outputs["Image"], output_node.inputs["Image"])

    arrange_nodes(group)

    # --------------------------------------------------------------------------

    group = bpy.data.node_groups.new(type="CompositorNodeTree", name="SplitTone")

    input_node = group.nodes.new("NodeGroupInput")

    group.inputs.new("NodeSocketColor", "Image")
    group.inputs.new("NodeSocketFloat", "HighlightsHue")
    group.inputs.new("NodeSocketFloat", "HighlightsSaturation")
    group.inputs.new("NodeSocketFloat", "ShadowsHue")
    group.inputs.new("NodeSocketFloat", "ShadowsSaturation")
    group.inputs.new("NodeSocketFloatFactor", "Balance")

    set_socket_value_range(group.inputs["HighlightsHue"])
    set_socket_value_range(group.inputs["HighlightsSaturation"])
    set_socket_value_range(group.inputs["ShadowsHue"])
    set_socket_value_range(group.inputs["ShadowsSaturation"])
    set_socket_value_range(group.inputs["Balance"], default_value=0.5)

    input_sep_node = group.nodes.new(type="CompositorNodeSepHSVA")

    subtract_node = group.nodes.new(type="CompositorNodeMath")
    subtract_node.inputs[0].default_value = 1.0
    subtract_node.operation = 'SUBTRACT'
    subtract_node.use_clamp = True

    multiply_node = group.nodes.new(type="CompositorNodeMath")
    multiply_node.inputs[1].default_value = 2.0
    multiply_node.operation = 'MULTIPLY'
    multiply_node.use_clamp = False

    power_node = group.nodes.new(type="CompositorNodeMath")
    power_node.operation = 'POWER'
    power_node.use_clamp = True

    shadows_node = group.nodes.new(type='CompositorNodeGroup')
    shadows_node.name = "Shadows"
    shadows_node.node_tree = bpy.data.node_groups["SplitToneSub"]

    highlights_node = group.nodes.new(type='CompositorNodeGroup')
    highlights_node.name = "Highlights"
    highlights_node.node_tree = bpy.data.node_groups["SplitToneSub"]

    comb_node = group.nodes.new(type="CompositorNodeMixRGB")
    comb_node.use_clamp = False

    output_node = group.nodes.new("NodeGroupOutput")
    group.outputs.new("NodeSocketColor", "Image")

    group.links.new(input_node.outputs["Image"], input_sep_node.inputs["Image"])
    group.links.new(input_node.outputs["Image"], shadows_node.inputs["Image"])
    group.links.new(input_node.outputs["ShadowsHue"], shadows_node.inputs["Hue"])
    group.links.new(input_node.outputs["ShadowsSaturation"], shadows_node.inputs["Saturation"])
    group.links.new(input_node.outputs["Image"], highlights_node.inputs["Image"])
    group.links.new(input_node.outputs["HighlightsHue"], highlights_node.inputs["Hue"])
    group.links.new(input_node.outputs["HighlightsSaturation"], highlights_node.inputs["Saturation"])
    group.links.new(input_node.outputs["Balance"], subtract_node.inputs[1])
    group.links.new(subtract_node.outputs["Value"], multiply_node.inputs[0])
    group.links.new(input_sep_node.outputs["V"], power_node.inputs[0])
    group.links.new(multiply_node.outputs["Value"], power_node.inputs[1])
    group.links.new(power_node.outputs["Value"], comb_node.inputs["Fac"])
    group.links.new(shadows_node.outputs["Image"], comb_node.inputs[1])
    group.links.new(highlights_node.outputs["Image"], comb_node.inputs[2])
    group.links.new(comb_node.outputs["Image"], output_node.inputs["Image"])

    arrange_nodes(group)

    return group


def add_vignette_node_group() -> bpy.types.NodeGroup:
    group = bpy.data.node_groups.new(type="CompositorNodeTree", name="Vignette")

    input_node = group.nodes.new("NodeGroupInput")
    group.inputs.new("NodeSocketColor", "Image")
    group.inputs.new("NodeSocketFloat", "Amount")
    group.inputs["Amount"].default_value = 0.2
    group.inputs["Amount"].min_value = 0.0
    group.inputs["Amount"].max_value = 1.0

    lens_distortion_node = group.nodes.new(type="CompositorNodeLensdist")
    lens_distortion_node.inputs["Distort"].default_value = 1.000

    separate_rgba_node = group.nodes.new(type="CompositorNodeSepRGBA")

    blur_node = group.nodes.new(type="CompositorNodeBlur")
    blur_node.filter_type = 'GAUSS'
    blur_node.size_x = 300
    blur_node.size_y = 300
    blur_node.use_extended_bounds = True

    mix_node = group.nodes.new(type="CompositorNodeMixRGB")
    mix_node.blend_type = 'MULTIPLY'

    output_node = group.nodes.new("NodeGroupOutput")
    group.outputs.new("NodeSocketColor", "Image")

    group.links.new(input_node.outputs["Amount"], mix_node.inputs["Fac"])
    group.links.new(input_node.outputs["Image"], mix_node.inputs[1])
    group.links.new(input_node.outputs["Image"], lens_distortion_node.inputs["Image"])
    group.links.new(lens_distortion_node.outputs["Image"], separate_rgba_node.inputs["Image"])
    group.links.new(separate_rgba_node.outputs["A"], blur_node.inputs["Image"])
    group.links.new(blur_node.outputs["Image"], mix_node.inputs[2])
    group.links.new(mix_node.outputs["Image"], output_node.inputs["Image"])

    arrange_nodes(group)

    return group


def create_split_tone_node(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    split_tone_node_group = add_split_tone_node_group()

    node = node_tree.nodes.new(type='CompositorNodeGroup')
    node.name = "SplitTone"
    node.node_tree = split_tone_node_group

    return node


def create_vignette_node(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    vignette_node_group = add_vignette_node_group()

    node = node_tree.nodes.new(type='CompositorNodeGroup')
    node.name = "Vignette"
    node.node_tree = vignette_node_group

    return node


def build_scene_composition(scene: bpy.types.Scene,
                            vignette: float = 0.20,
                            dispersion: float = 0.050,
                            gain: float = 1.10,
                            saturation: float = 1.10) -> None:
    scene.use_nodes = True
    clean_nodes(scene.node_tree.nodes)

    render_layer_node = scene.node_tree.nodes.new(type="CompositorNodeRLayers")

    vignette_node = create_vignette_node(scene.node_tree)
    vignette_node.inputs["Amount"].default_value = vignette

    lens_distortion_node = scene.node_tree.nodes.new(type="CompositorNodeLensdist")
    lens_distortion_node.inputs["Distort"].default_value = -dispersion * 0.40
    lens_distortion_node.inputs["Dispersion"].default_value = dispersion

    color_correction_node = scene.node_tree.nodes.new(type="CompositorNodeColorCorrection")
    color_correction_node.master_saturation = saturation
    color_correction_node.master_gain = gain

    split_tone_node = create_split_tone_node(scene.node_tree)

    glare_node = scene.node_tree.nodes.new(type="CompositorNodeGlare")
    glare_node.glare_type = 'FOG_GLOW'
    glare_node.quality = 'HIGH'

    composite_node = scene.node_tree.nodes.new(type="CompositorNodeComposite")

    scene.node_tree.links.new(render_layer_node.outputs['Image'], vignette_node.inputs['Image'])
    scene.node_tree.links.new(vignette_node.outputs['Image'], lens_distortion_node.inputs['Image'])
    scene.node_tree.links.new(lens_distortion_node.outputs['Image'], color_correction_node.inputs['Image'])
    scene.node_tree.links.new(color_correction_node.outputs['Image'], split_tone_node.inputs['Image'])
    scene.node_tree.links.new(split_tone_node.outputs['Image'], glare_node.inputs['Image'])
    scene.node_tree.links.new(glare_node.outputs['Image'], composite_node.inputs['Image'])

    arrange_nodes(scene.node_tree)


# =============================================================================
# COMPOSITION END
# =============================================================================


# =============================================================================
# IMAGE 
# =============================================================================


def get_image_pixels_in_numpy(image: bpy.types.Image) -> np.array:
    return np.array(image.pixels[:]).reshape(image.size[0] * image.size[1], image.channels)


def set_image_pixels_in_numpy(image: bpy.types.Image, pixels: np.array) -> None:
    # The sizes should be the same; otherwise, Blender may crush.
    assert len(image.pixels) == pixels.size

    image.pixels = pixels.flatten()


# =============================================================================
# IMAGE END
# =============================================================================


# =============================================================================
# LIGHTING
# =============================================================================

def create_area_light(location: Tuple[float, float, float] = (0.0, 0.0, 5.0),
                      rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      size: float = 5.0,
                      color: Tuple[float, float, float, float] = (1.00, 0.90, 0.80, 1.00),
                      strength: float = 1000.0,
                      name: Optional[str] = None) -> bpy.types.Object:
    if bpy.app.version >= (2, 80, 0):
        bpy.ops.object.light_add(type='AREA', location=location, rotation=rotation)
    else:
        bpy.ops.object.lamp_add(type='AREA', location=location, rotation=rotation)

    if name is not None:
        bpy.context.object.name = name

    light = bpy.context.object.data
    light.size = size
    light.use_nodes = True
    light.node_tree.nodes["Emission"].inputs["Color"].default_value = color
    light.energy = strength

    return bpy.context.object


def create_sun_light(location: Tuple[float, float, float] = (0.0, 0.0, 5.0),
                     rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     name: Optional[str] = None) -> bpy.types.Object:
    bpy.ops.object.light_add(type='SUN', location=location, rotation=rotation)

    if name is not None:
        bpy.context.object.name = name

    return bpy.context.object


# =============================================================================
# LIGHTING END
# =============================================================================


# =============================================================================
# MATERIAL
# =============================================================================


def create_texture_node(node_tree: bpy.types.NodeTree, path: str, is_color_data: bool) -> bpy.types.Node:
    # Instantiate a new texture image node
    texture_node = node_tree.nodes.new(type='ShaderNodeTexImage')

    # Open an image and set it to the node
    texture_node.image = bpy.data.images.load(path)

    # Set other parameters
    texture_node.image.colorspace_settings.is_data = False if is_color_data else True

    # Return the node
    return texture_node


def set_principled_node(principled_node: bpy.types.Node,
                        base_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
                        subsurface: float = 0.0,
                        subsurface_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
                        subsurface_radius: Tuple[float, float, float] = (1.0, 0.2, 0.1),
                        metallic: float = 0.0,
                        specular: float = 0.5,
                        specular_tint: float = 0.0,
                        roughness: float = 0.5,
                        anisotropic: float = 0.0,
                        anisotropic_rotation: float = 0.0,
                        sheen: float = 0.0,
                        sheen_tint: float = 0.5,
                        clearcoat: float = 0.0,
                        clearcoat_roughness: float = 0.03,
                        ior: float = 1.45,
                        transmission: float = 0.0,
                        transmission_roughness: float = 0.0) -> None:
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Subsurface'].default_value = subsurface
    principled_node.inputs['Subsurface Color'].default_value = subsurface_color
    principled_node.inputs['Subsurface Radius'].default_value = subsurface_radius
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Specular'].default_value = specular
    principled_node.inputs['Specular Tint'].default_value = specular_tint
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Anisotropic'].default_value = anisotropic
    principled_node.inputs['Anisotropic Rotation'].default_value = anisotropic_rotation
    principled_node.inputs['Sheen'].default_value = sheen
    principled_node.inputs['Sheen Tint'].default_value = sheen_tint
    principled_node.inputs['Clearcoat'].default_value = clearcoat
    principled_node.inputs['Clearcoat Roughness'].default_value = clearcoat_roughness
    principled_node.inputs['IOR'].default_value = ior
    principled_node.inputs['Transmission'].default_value = transmission
    principled_node.inputs['Transmission Roughness'].default_value = transmission_roughness


def build_pbr_nodes(node_tree: bpy.types.NodeTree,
                    base_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
                    metallic: float = 0.0,
                    specular: float = 0.5,
                    roughness: float = 0.5,
                    sheen: float = 0.0) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    set_principled_node(principled_node=principled_node,
                        base_color=base_color,
                        metallic=metallic,
                        specular=specular,
                        roughness=roughness,
                        sheen=sheen)

    arrange_nodes(node_tree)


def build_checker_board_nodes(node_tree: bpy.types.NodeTree, size: float) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    checker_texture_node = node_tree.nodes.new(type='ShaderNodeTexChecker')

    set_principled_node(principled_node=principled_node)
    checker_texture_node.inputs['Scale'].default_value = size

    node_tree.links.new(checker_texture_node.outputs['Color'], principled_node.inputs['Base Color'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    arrange_nodes(node_tree)


def build_matcap_nodes(node_tree: bpy.types.NodeTree, image_path: str) -> None:
    tex_coord_node = node_tree.nodes.new(type='ShaderNodeTexCoord')
    vector_transform_node = node_tree.nodes.new(type='ShaderNodeVectorTransform')
    mapping_node = node_tree.nodes.new(type='ShaderNodeMapping')
    texture_image_node = create_texture_node(node_tree, image_path, True)
    emmission_node = node_tree.nodes.new(type='ShaderNodeEmission')
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    create_frame_node(node_tree, (tex_coord_node, vector_transform_node, mapping_node),
                      name="MatCap UV",
                      label="MatCap UV")

    vector_transform_node.vector_type = "VECTOR"
    vector_transform_node.convert_from = "OBJECT"
    vector_transform_node.convert_to = "CAMERA"

    mapping_node.vector_type = "TEXTURE"
    if bpy.app.version >= (2, 81, 0):
        mapping_node.inputs["Location"].default_value = (1.0, 1.0, 0.0)
        mapping_node.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
    else:
        mapping_node.translation = (1.0, 1.0, 0.0)
        mapping_node.scale = (2.0, 2.0, 1.0)

    node_tree.links.new(tex_coord_node.outputs['Normal'], vector_transform_node.inputs['Vector'])
    node_tree.links.new(vector_transform_node.outputs['Vector'], mapping_node.inputs['Vector'])
    node_tree.links.new(mapping_node.outputs['Vector'], texture_image_node.inputs['Vector'])
    node_tree.links.new(texture_image_node.outputs['Color'], emmission_node.inputs['Color'])
    node_tree.links.new(emmission_node.outputs['Emission'], output_node.inputs['Surface'])

    arrange_nodes(node_tree)


def build_pbr_textured_nodes(node_tree: bpy.types.NodeTree,
                             color_texture_path: str = "",
                             metallic_texture_path: str = "",
                             roughness_texture_path: str = "",
                             normal_texture_path: str = "",
                             displacement_texture_path: str = "",
                             ambient_occlusion_texture_path: str = "",
                             scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                             displacement_scale: float = 1.0) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    coord_node = node_tree.nodes.new(type='ShaderNodeTexCoord')
    mapping_node = node_tree.nodes.new(type='ShaderNodeMapping')
    mapping_node.vector_type = 'TEXTURE'
    if bpy.app.version >= (2, 81, 0):
        mapping_node.inputs["Scale"].default_value = scale
    else:
        mapping_node.scale = scale
    node_tree.links.new(coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    if color_texture_path != "":
        texture_node = create_texture_node(node_tree, color_texture_path, True)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        if ambient_occlusion_texture_path != "":
            ao_texture_node = create_texture_node(node_tree, ambient_occlusion_texture_path, False)
            node_tree.links.new(mapping_node.outputs['Vector'], ao_texture_node.inputs['Vector'])
            mix_node = node_tree.nodes.new(type='ShaderNodeMixRGB')
            mix_node.blend_type = 'MULTIPLY'
            node_tree.links.new(texture_node.outputs['Color'], mix_node.inputs['Color1'])
            node_tree.links.new(ao_texture_node.outputs['Color'], mix_node.inputs['Color2'])
            node_tree.links.new(mix_node.outputs['Color'], principled_node.inputs['Base Color'])
        else:
            node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])

    if metallic_texture_path != "":
        texture_node = create_texture_node(node_tree, metallic_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Metallic'])

    if roughness_texture_path != "":
        texture_node = create_texture_node(node_tree, roughness_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Roughness'])

    if normal_texture_path != "":
        texture_node = create_texture_node(node_tree, normal_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        normal_map_node = node_tree.nodes.new(type='ShaderNodeNormalMap')
        node_tree.links.new(texture_node.outputs['Color'], normal_map_node.inputs['Color'])
        node_tree.links.new(normal_map_node.outputs['Normal'], principled_node.inputs['Normal'])

    if displacement_texture_path != "":
        texture_node = create_texture_node(node_tree, displacement_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        displacement_node = node_tree.nodes.new(type='ShaderNodeDisplacement')
        displacement_node.inputs['Scale'].default_value = displacement_scale
        node_tree.links.new(texture_node.outputs['Color'], displacement_node.inputs['Height'])
        node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])

    arrange_nodes(node_tree)


def add_parametric_color_ramp() -> bpy.types.NodeGroup:
    group = bpy.data.node_groups.new(type="ShaderNodeTree", name="Parametric Color Ramp")

    # Input

    input_node = group.nodes.new(type="NodeGroupInput")
    group.inputs.new("NodeSocketFloatFactor", "Fac")
    group.inputs.new("NodeSocketColor", "Color1")
    group.inputs.new("NodeSocketColor", "Color2")
    group.inputs.new("NodeSocketFloatFactor", "Pos1")
    group.inputs.new("NodeSocketFloatFactor", "Pos2")

    set_socket_value_range(group.inputs["Fac"], default_value=0.5)
    set_socket_value_range(group.inputs["Pos1"], default_value=0.0)
    set_socket_value_range(group.inputs["Pos2"], default_value=1.0)

    # Math

    denominator_subtract_node = group.nodes.new(type="ShaderNodeMath")
    denominator_subtract_node.operation = "SUBTRACT"
    denominator_subtract_node.use_clamp = True

    numerator_subtract_node = group.nodes.new(type="ShaderNodeMath")
    numerator_subtract_node.operation = "SUBTRACT"
    numerator_subtract_node.use_clamp = True

    divide_node = group.nodes.new(type="ShaderNodeMath")
    divide_node.operation = "DIVIDE"
    divide_node.use_clamp = True

    group.links.new(input_node.outputs["Pos2"], denominator_subtract_node.inputs[0])
    group.links.new(input_node.outputs["Fac"], denominator_subtract_node.inputs[1])

    group.links.new(input_node.outputs["Pos2"], numerator_subtract_node.inputs[0])
    group.links.new(input_node.outputs["Pos1"], numerator_subtract_node.inputs[1])

    group.links.new(denominator_subtract_node.outputs["Value"], divide_node.inputs[0])
    group.links.new(numerator_subtract_node.outputs["Value"], divide_node.inputs[1])

    # Mixing

    mix_node = group.nodes.new(type="ShaderNodeMixRGB")

    group.links.new(divide_node.outputs["Value"], mix_node.inputs["Fac"])
    group.links.new(input_node.outputs["Color2"], mix_node.inputs[1])
    group.links.new(input_node.outputs["Color1"], mix_node.inputs[2])

    # Output

    output_node = group.nodes.new(type="NodeGroupOutput")
    group.outputs.new("NodeSocketColor", "Color")

    group.links.new(mix_node.outputs["Color"], output_node.inputs["Color"])

    # Return

    arrange_nodes(group)

    return group


def create_parametric_color_ramp_node(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    color_ramp_node_group: bpy.types.NodeGroup

    if "Parametric Color Ramp" in bpy.data.node_groups:
        color_ramp_node_group = bpy.data.node_groups["Parametric Color Ramp"]
    else:
        color_ramp_node_group = add_parametric_color_ramp()

    node = node_tree.nodes.new(type='ShaderNodeGroup')
    node.name = "Parametric Color Ramp"
    node.node_tree = color_ramp_node_group

    return node


def add_tri_parametric_color_ramp() -> bpy.types.NodeGroup:
    group = bpy.data.node_groups.new(type="ShaderNodeTree", name="Tri Parametric Color Ramp")

    # Input

    input_node = group.nodes.new(type="NodeGroupInput")
    group.inputs.new("NodeSocketFloatFactor", "Fac")
    group.inputs.new("NodeSocketColor", "Color1")
    group.inputs.new("NodeSocketColor", "Color2")
    group.inputs.new("NodeSocketColor", "Color3")
    group.inputs.new("NodeSocketFloatFactor", "Pos1")
    group.inputs.new("NodeSocketFloatFactor", "Pos2")
    group.inputs.new("NodeSocketFloatFactor", "Pos3")

    set_socket_value_range(group.inputs["Fac"], default_value=0.5)
    set_socket_value_range(group.inputs["Pos1"], default_value=0.25)
    set_socket_value_range(group.inputs["Pos2"], default_value=0.50)
    set_socket_value_range(group.inputs["Pos3"], default_value=0.75)

    # Nested color ramp

    nested_color_ramp_node = create_parametric_color_ramp_node(group)

    group.links.new(input_node.outputs["Color1"], nested_color_ramp_node.inputs["Color1"])
    group.links.new(input_node.outputs["Color2"], nested_color_ramp_node.inputs["Color2"])
    group.links.new(input_node.outputs["Pos1"], nested_color_ramp_node.inputs["Pos1"])
    group.links.new(input_node.outputs["Pos2"], nested_color_ramp_node.inputs["Pos2"])
    group.links.new(input_node.outputs["Fac"], nested_color_ramp_node.inputs["Fac"])

    # Math

    denominator_subtract_node = group.nodes.new(type="ShaderNodeMath")
    denominator_subtract_node.operation = "SUBTRACT"
    denominator_subtract_node.use_clamp = True

    numerator_subtract_node = group.nodes.new(type="ShaderNodeMath")
    numerator_subtract_node.operation = "SUBTRACT"
    numerator_subtract_node.use_clamp = True

    divide_node = group.nodes.new(type="ShaderNodeMath")
    divide_node.operation = "DIVIDE"
    divide_node.use_clamp = True

    group.links.new(input_node.outputs["Pos3"], denominator_subtract_node.inputs[0])
    group.links.new(input_node.outputs["Fac"], denominator_subtract_node.inputs[1])

    group.links.new(input_node.outputs["Pos3"], numerator_subtract_node.inputs[0])
    group.links.new(input_node.outputs["Pos2"], numerator_subtract_node.inputs[1])

    group.links.new(denominator_subtract_node.outputs["Value"], divide_node.inputs[0])
    group.links.new(numerator_subtract_node.outputs["Value"], divide_node.inputs[1])

    # Mixing

    mix_node = group.nodes.new(type="ShaderNodeMixRGB")

    group.links.new(divide_node.outputs["Value"], mix_node.inputs["Fac"])
    group.links.new(input_node.outputs["Color3"], mix_node.inputs[1])
    group.links.new(nested_color_ramp_node.outputs["Color"], mix_node.inputs[2])

    # Output

    output_node = group.nodes.new(type="NodeGroupOutput")
    group.outputs.new("NodeSocketColor", "Color")

    group.links.new(mix_node.outputs["Color"], output_node.inputs["Color"])

    # Return

    arrange_nodes(group)

    return group


def create_tri_parametric_color_ramp_node(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    tri_color_ramp_node_group: bpy.types.NodeGroup

    if "Tri Parametric Color Ramp" in bpy.data.node_groups:
        tri_color_ramp_node_group = bpy.data.node_groups["Tri Parametric Color Ramp"]
    else:
        tri_color_ramp_node_group = add_tri_parametric_color_ramp()

    node = node_tree.nodes.new(type='ShaderNodeGroup')
    node.name = "Tri Parametric Color Ramp"
    node.node_tree = tri_color_ramp_node_group

    return node


def add_peeling_paint_metal_node_group() -> bpy.types.NodeGroup:
    group = bpy.data.node_groups.new(type="ShaderNodeTree", name="Peeling Paint Metal")

    input_node = group.nodes.new(type="NodeGroupInput")
    group.inputs.new("NodeSocketColor", "Paint Color")
    group.inputs.new("NodeSocketFloatFactor", "Paint Roughness")
    group.inputs.new("NodeSocketColor", "Metal Color")
    group.inputs.new("NodeSocketFloatFactor", "Metal Roughness")
    group.inputs.new("NodeSocketFloat", "Scale")
    group.inputs.new("NodeSocketFloat", "Detail")
    group.inputs.new("NodeSocketFloat", "Distortion")
    group.inputs.new("NodeSocketFloatFactor", "Threshold")
    group.inputs.new("NodeSocketFloat", "Peel Intense")

    group.inputs["Paint Color"].default_value = (0.152, 0.524, 0.067, 1.000)
    group.inputs["Metal Color"].default_value = (0.062, 0.015, 0.011, 1.000)

    set_socket_value_range(group.inputs["Paint Roughness"], default_value=0.05)
    set_socket_value_range(group.inputs["Metal Roughness"], default_value=0.50)

    set_socket_value_range(group.inputs["Scale"], default_value=4.5, min_value=0.0, max_value=1000.0)
    set_socket_value_range(group.inputs["Detail"], default_value=8.0, min_value=0.0, max_value=16.0)
    set_socket_value_range(group.inputs["Distortion"], default_value=0.5, min_value=0.0, max_value=1000.0)
    set_socket_value_range(group.inputs["Threshold"], default_value=0.42)
    set_socket_value_range(group.inputs["Peel Intense"], default_value=0.2, min_value=0.0, max_value=1.0)

    tex_coord_node = group.nodes.new(type="ShaderNodeTexCoord")
    mapping_node = group.nodes.new(type="ShaderNodeMapping")

    group.links.new(tex_coord_node.outputs["Object"], mapping_node.inputs["Vector"])

    # Peeling region segmentation

    peeling_noise_node = group.nodes.new(type="ShaderNodeTexNoise")

    group.links.new(mapping_node.outputs["Vector"], peeling_noise_node.inputs["Vector"])
    group.links.new(input_node.outputs["Scale"], peeling_noise_node.inputs["Scale"])
    group.links.new(input_node.outputs["Detail"], peeling_noise_node.inputs["Detail"])
    group.links.new(input_node.outputs["Distortion"], peeling_noise_node.inputs["Distortion"])

    peeling_threshold_node = create_parametric_color_ramp_node(group)
    peeling_threshold_node.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)
    peeling_threshold_node.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)

    # Base color

    epsilon_subtract_node = group.nodes.new(type="ShaderNodeMath")
    epsilon_subtract_node.operation = "SUBTRACT"
    epsilon_subtract_node.inputs[1].default_value = 0.001

    group.links.new(input_node.outputs["Threshold"], epsilon_subtract_node.inputs[0])

    group.links.new(peeling_noise_node.outputs["Fac"], peeling_threshold_node.inputs["Fac"])
    group.links.new(epsilon_subtract_node.outputs["Value"], peeling_threshold_node.inputs["Pos1"])
    group.links.new(input_node.outputs["Threshold"], peeling_threshold_node.inputs["Pos2"])

    color_mix_node = group.nodes.new(type="ShaderNodeMixRGB")
    group.links.new(peeling_threshold_node.outputs["Color"], color_mix_node.inputs["Fac"])
    group.links.new(input_node.outputs["Metal Color"], color_mix_node.inputs[1])
    group.links.new(input_node.outputs["Paint Color"], color_mix_node.inputs[2])

    # Ambient Occlusion

    epsilon_add_node = group.nodes.new(type="ShaderNodeMath")
    epsilon_add_node.operation = "ADD"
    epsilon_add_node.inputs[1].default_value = 0.010

    group.links.new(input_node.outputs["Threshold"], epsilon_add_node.inputs[0])

    fallout_subtract_node = group.nodes.new(type="ShaderNodeMath")
    fallout_subtract_node.operation = "SUBTRACT"
    fallout_subtract_node.inputs[1].default_value = 0.060

    group.links.new(input_node.outputs["Threshold"], fallout_subtract_node.inputs[0])

    ao_node = create_tri_parametric_color_ramp_node(group)
    ao_node.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    ao_node.inputs["Color2"].default_value = (0.0, 0.0, 0.0, 1.0)
    ao_node.inputs["Color3"].default_value = (1.0, 1.0, 1.0, 1.0)

    group.links.new(peeling_noise_node.outputs["Fac"], ao_node.inputs["Fac"])
    group.links.new(fallout_subtract_node.outputs["Value"], ao_node.inputs["Pos1"])
    group.links.new(input_node.outputs["Threshold"], ao_node.inputs["Pos2"])
    group.links.new(epsilon_add_node.outputs["Value"], ao_node.inputs["Pos3"])

    ao_mix_node = group.nodes.new(type="ShaderNodeMixRGB")
    ao_mix_node.blend_type = "MULTIPLY"
    ao_mix_node.inputs["Fac"].default_value = 1.0

    group.links.new(color_mix_node.outputs["Color"], ao_mix_node.inputs[1])
    group.links.new(ao_node.outputs["Color"], ao_mix_node.inputs[2])

    create_frame_node(group, nodes=(epsilon_add_node, fallout_subtract_node, ao_node), name="AO", label="AO")

    # Metallic

    metallic_node = group.nodes.new(type="ShaderNodeMixRGB")
    metallic_node.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    metallic_node.inputs["Color2"].default_value = (0.0, 0.0, 0.0, 1.0)

    group.links.new(peeling_threshold_node.outputs["Color"], metallic_node.inputs["Fac"])

    # Roughness

    roughness_node = group.nodes.new(type="ShaderNodeMixRGB")

    group.links.new(input_node.outputs["Metal Roughness"], roughness_node.inputs["Color1"])
    group.links.new(input_node.outputs["Paint Roughness"], roughness_node.inputs["Color2"])
    group.links.new(peeling_threshold_node.outputs["Color"], roughness_node.inputs["Fac"])

    # Bump

    height_node = create_tri_parametric_color_ramp_node(group)
    height_node.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)
    height_node.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    height_node.inputs["Color3"].default_value = (0.5, 0.5, 0.5, 1.0)

    height_peak_add_node = group.nodes.new(type="ShaderNodeMath")
    height_peak_add_node.operation = "MULTIPLY_ADD"
    height_peak_add_node.inputs[1].default_value = 0.025
    height_peak_add_node.label = "Height Peak Add"

    height_tail_add_node = group.nodes.new(type="ShaderNodeMath")
    height_tail_add_node.operation = "MULTIPLY_ADD"
    height_tail_add_node.inputs[1].default_value = 0.100
    height_tail_add_node.label = "Height Tail Add"

    group.links.new(input_node.outputs["Threshold"], height_peak_add_node.inputs[2])
    group.links.new(input_node.outputs["Peel Intense"], height_peak_add_node.inputs[0])
    group.links.new(height_peak_add_node.outputs["Value"], height_tail_add_node.inputs[2])
    group.links.new(input_node.outputs["Peel Intense"], height_tail_add_node.inputs[0])
    group.links.new(peeling_noise_node.outputs["Fac"], height_node.inputs["Fac"])
    group.links.new(input_node.outputs["Threshold"], height_node.inputs["Pos1"])
    group.links.new(height_peak_add_node.outputs["Value"], height_node.inputs["Pos2"])
    group.links.new(height_tail_add_node.outputs["Value"], height_node.inputs["Pos3"])

    bump_node = group.nodes.new(type="ShaderNodeBump")
    group.links.new(height_node.outputs["Color"], bump_node.inputs["Height"])

    create_frame_node(group,
                      nodes=(height_node, height_peak_add_node, height_tail_add_node, bump_node),
                      name="Bump",
                      label="Bump")

    # Output

    output_node = group.nodes.new("NodeGroupOutput")
    group.outputs.new("NodeSocketColor", "Color")
    group.outputs.new("NodeSocketFloatFactor", "Metallic")
    group.outputs.new("NodeSocketFloatFactor", "Roughness")
    group.outputs.new("NodeSocketVectorDirection", "Bump")

    group.links.new(ao_mix_node.outputs["Color"], output_node.inputs["Color"])
    group.links.new(metallic_node.outputs["Color"], output_node.inputs["Metallic"])
    group.links.new(roughness_node.outputs["Color"], output_node.inputs["Roughness"])
    group.links.new(bump_node.outputs["Normal"], output_node.inputs["Bump"])

    arrange_nodes(group)

    return group


def create_peeling_paint_metal_node_group(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    peeling_paint_metal_node_group: bpy.types.NodeGroup

    if "Peeling Paint Metal" in bpy.data.node_groups:
        peeling_paint_metal_node_group = bpy.data.node_groups["Peeling Paint Metal"]
    else:
        peeling_paint_metal_node_group = add_peeling_paint_metal_node_group()

    node = node_tree.nodes.new(type='ShaderNodeGroup')
    node.name = "Peeling Paint Metal"
    node.node_tree = peeling_paint_metal_node_group

    return node


def build_peeling_paint_metal_nodes(node_tree: bpy.types.NodeTree) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    peeling_paint_metal_node = create_peeling_paint_metal_node_group(node_tree)

    node_tree.links.new(peeling_paint_metal_node.outputs['Color'], principled_node.inputs['Base Color'])
    node_tree.links.new(peeling_paint_metal_node.outputs['Metallic'], principled_node.inputs['Metallic'])
    node_tree.links.new(peeling_paint_metal_node.outputs['Roughness'], principled_node.inputs['Roughness'])
    node_tree.links.new(peeling_paint_metal_node.outputs['Bump'], principled_node.inputs['Normal'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    arrange_nodes(node_tree)


def build_emission_nodes(node_tree: bpy.types.NodeTree,
                         color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         strength: float = 1.0) -> None:
    """
    https://docs.blender.org/api/current/bpy.types.ShaderNodeEmission.html
    """
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = node_tree.nodes.new(type='ShaderNodeEmission')

    emission_node.inputs["Color"].default_value = color + (1.0,)
    emission_node.inputs["Strength"].default_value = strength

    node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    arrange_nodes(node_tree)


def add_material(name: str = "Material",
                 use_nodes: bool = False,
                 make_node_tree_empty: bool = False) -> bpy.types.Material:
    """
    https://docs.blender.org/api/current/bpy.types.BlendDataMaterials.html
    https://docs.blender.org/api/current/bpy.types.Material.html
    """

    # TODO: Check whether the name is already used or not

    material = bpy.data.materials.new(name)
    material.use_nodes = use_nodes

    if use_nodes and make_node_tree_empty:
        clean_nodes(material.node_tree.nodes)

    return material


# =============================================================================
# MATERIAL END
# =============================================================================


# =============================================================================
# MESH
# =============================================================================


def set_smooth_shading(mesh: bpy.types.Mesh) -> None:
    for polygon in mesh.polygons:
        polygon.use_smooth = True


def create_mesh_from_pydata(scene: bpy.types.Scene,
                            vertices: Iterable[Iterable[float]],
                            faces: Iterable[Iterable[int]],
                            mesh_name: str,
                            object_name: str,
                            use_smooth: bool = True) -> bpy.types.Object:
    # Add a new mesh and set vertices and faces
    # In this case, it does not require to set edges
    # After manipulating mesh data, update() needs to be called
    new_mesh: bpy.types.Mesh = bpy.data.meshes.new(mesh_name)
    new_mesh.from_pydata(vertices, [], faces)
    new_mesh.update()
    if use_smooth:
        set_smooth_shading(new_mesh)

    new_object: bpy.types.Object = bpy.data.objects.new(object_name, new_mesh)
    scene.collection.objects.link(new_object)

    return new_object


def create_cached_mesh_from_alembic(file_path: str, name: str) -> bpy.types.Object:
    bpy.ops.wm.alembic_import(filepath=file_path, as_background_job=False)
    bpy.context.active_object.name = name

    return bpy.context.active_object


def create_plane(location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 size: float = 2.0,
                 name: Optional[str] = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)

    current_object = bpy.context.object

    if name is not None:
        current_object.name = name

    return current_object


def create_smooth_sphere(location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         radius: float = 1.0,
                         subdivision_level: int = 1,
                         name: Optional[str] = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, calc_uvs=True)

    current_object = bpy.context.object

    if name is not None:
        current_object.name = name

    set_smooth_shading(current_object.data)
    add_subdivision_surface_modifier(current_object, subdivision_level)

    return current_object


def create_smooth_monkey(location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         subdivision_level: int = 2,
                         name: Optional[str] = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_monkey_add(location=location, rotation=rotation, calc_uvs=True)

    current_object = bpy.context.object

    if name is not None:
        current_object.name = name

    set_smooth_shading(current_object.data)
    add_subdivision_surface_modifier(current_object, subdivision_level)

    return current_object


def create_three_smooth_monkeys(
        names: Optional[Tuple[str, str, str]] = None) -> Tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    if names is None:
        names = ("Suzanne Left", "Suzanne Center", "Suzanne Right")

    left = create_smooth_monkey(location=(-1.8, 0.0, 1.0), rotation=(0.0, 0.0, -math.pi * 60.0 / 180.0), name=names[0])
    center = create_smooth_monkey(location=(0.0, 0.0, 1.0), rotation=(0.0, 0.0, -math.pi * 60.0 / 180.0), name=names[1])
    right = create_smooth_monkey(location=(+1.8, 0.0, 1.0), rotation=(0.0, 0.0, -math.pi * 60.0 / 180.0), name=names[2])

    return left, center, right


# https://docs.blender.org/api/current/bpy.types.VertexGroups.html
# https://docs.blender.org/api/current/bpy.types.VertexGroup.html
def add_vertex_group(mesh_object: bpy.types.Object, name: str = "Group") -> bpy.types.VertexGroup:
    # TODO: Check whether the object has a mesh data
    # TODO: Check whether the object already has a vertex group with the specified name

    vertex_group = mesh_object.vertex_groups.new(name=name)

    return vertex_group


# =============================================================================
# MESH END
# =============================================================================


# =============================================================================
# MODIFIER
# =============================================================================


def add_boolean_modifier(mesh_object: bpy.types.Object,
                         another_mesh_object: bpy.types.Object,
                         operation: str = "DIFFERENCE") -> None:
    """
    https://docs.blender.org/api/current/bpy.types.BooleanModifier.html
    """

    modifier: bpy.types.SubsurfModifier = mesh_object.modifiers.new(name="Boolean", type='BOOLEAN')

    modifier.object = another_mesh_object
    modifier.operation = operation


def add_subdivision_surface_modifier(mesh_object: bpy.types.Object, level: int, is_simple: bool = False) -> None:
    """
    https://docs.blender.org/api/current/bpy.types.SubsurfModifier.html
    """

    modifier: bpy.types.SubsurfModifier = mesh_object.modifiers.new(name="Subsurf", type='SUBSURF')

    modifier.levels = level
    modifier.render_levels = level
    modifier.subdivision_type = 'SIMPLE' if is_simple else 'CATMULL_CLARK'


def add_solidify_modifier(mesh_object: bpy.types.Object,
                          thickness: float = 0.01,
                          flip_normal: bool = False,
                          fill_rim: bool = True,
                          material_index_offset: int = 0,
                          shell_vertex_group: str = "",
                          rim_vertex_group: str = "") -> None:
    """
    https://docs.blender.org/api/current/bpy.types.SolidifyModifier.html
    """

    modifier: bpy.types.SolidifyModifier = mesh_object.modifiers.new(name="Solidify", type='SOLIDIFY')

    modifier.material_offset = material_index_offset
    modifier.thickness = thickness
    modifier.use_flip_normals = flip_normal
    modifier.use_rim = fill_rim

    # TODO: Check whether shell_vertex_group is either empty or defined
    # TODO: Check whether rim_vertex_group is either empty or defined

    modifier.shell_vertex_group = shell_vertex_group
    modifier.rim_vertex_group = rim_vertex_group


def add_displace_modifier(mesh_object: bpy.types.Object,
                          texture_name: str,
                          vertex_group: str = "",
                          mid_level: float = 0.5,
                          strength: float = 1.0) -> None:
    """
    https://docs.blender.org/api/current/bpy.types.DisplaceModifier.html
    """

    modifier = mesh_object.modifiers.new(name="Displace", type='DISPLACE')

    modifier.mid_level = mid_level
    modifier.strength = strength

    # TODO: Check whether texture_name is properly defined
    modifier.texture = bpy.data.textures[texture_name]

    # TODO: Check whether vertex_group is either empty or defined
    modifier.vertex_group = vertex_group


# =============================================================================
# MODIFIER END
# =============================================================================


# =============================================================================
# NODE
# =============================================================================


def create_frame_node(node_tree: bpy.types.NodeTree,
                      nodes: Iterable[bpy.types.Node] = [],
                      name: str = "Frame",
                      label: str = "Frame") -> bpy.types.Node:
    frame_node = node_tree.nodes.new(type='NodeFrame')
    frame_node.name = name
    frame_node.label = label

    for node in nodes:
        node.parent = frame_node

    return frame_node


def set_socket_value_range(socket: bpy.types.NodeSocket,
                           default_value: float = 0.0,
                           min_value: float = 0.0,
                           max_value: float = 1.0) -> None:
    assert socket.type == "VALUE"

    socket.default_value = default_value
    socket.min_value = min_value
    socket.max_value = max_value


def clean_nodes(nodes: bpy.types.Nodes) -> None:
    for node in nodes:
        nodes.remove(node)


def arrange_nodes(node_tree: bpy.types.NodeTree, verbose: bool = False) -> None:
    max_num_iters = 2000
    epsilon = 1e-05
    target_space = 50.0

    second_stage = False

    fix_horizontal_location = True
    fix_vertical_location = True
    fix_overlaps = True

    if verbose:
        print("-----------------")
        print("Target nodes:")
        for node in node_tree.nodes:
            print("- " + node.name)

    # In the first stage, expand nodes overly
    target_space *= 2.0

    # Gauss-Seidel-style iterations
    previous_squared_deltas_sum = sys.float_info.max
    for i in range(max_num_iters):
        squared_deltas_sum = 0.0

        if fix_horizontal_location:
            for link in node_tree.links:
                k = 0.9 if not second_stage else 0.5
                threshold_factor = 2.0

                x_from = link.from_node.location[0]
                x_to = link.to_node.location[0]
                w_from = link.from_node.width
                signed_space = x_to - x_from - w_from
                C = signed_space - target_space
                grad_C_x_from = -1.0
                grad_C_x_to = 1.0

                # Skip if the distance is sufficiently large
                if C >= target_space * threshold_factor:
                    continue

                lagrange = C / (grad_C_x_from * grad_C_x_from + grad_C_x_to * grad_C_x_to)
                delta_x_from = -lagrange * grad_C_x_from
                delta_x_to = -lagrange * grad_C_x_to

                link.from_node.location[0] += k * delta_x_from
                link.to_node.location[0] += k * delta_x_to

                squared_deltas_sum += k * k * (delta_x_from * delta_x_from + delta_x_to * delta_x_to)

        if fix_vertical_location:
            k = 0.5 if not second_stage else 0.05
            socket_offset = 20.0

            def get_from_socket_index(node: bpy.types.Node, node_socket: bpy.types.NodeSocket) -> int:
                for i in range(len(node.outputs)):
                    if node.outputs[i] == node_socket:
                        return i
                assert False

            def get_to_socket_index(node: bpy.types.Node, node_socket: bpy.types.NodeSocket) -> int:
                for i in range(len(node.inputs)):
                    if node.inputs[i] == node_socket:
                        return i
                assert False

            for link in node_tree.links:
                from_socket_index = get_from_socket_index(link.from_node, link.from_socket)
                to_socket_index = get_to_socket_index(link.to_node, link.to_socket)
                y_from = link.from_node.location[1] - socket_offset * from_socket_index
                y_to = link.to_node.location[1] - socket_offset * to_socket_index
                C = y_from - y_to
                grad_C_y_from = 1.0
                grad_C_y_to = -1.0
                lagrange = C / (grad_C_y_from * grad_C_y_from + grad_C_y_to * grad_C_y_to)
                delta_y_from = -lagrange * grad_C_y_from
                delta_y_to = -lagrange * grad_C_y_to

                link.from_node.location[1] += k * delta_y_from
                link.to_node.location[1] += k * delta_y_to

                squared_deltas_sum += k * k * (delta_y_from * delta_y_from + delta_y_to * delta_y_to)

        if fix_overlaps and second_stage:
            k = 0.9
            margin = 0.5 * target_space

            # Examine all node pairs
            for node_1 in node_tree.nodes:
                for node_2 in node_tree.nodes:
                    if node_1 == node_2:
                        continue

                    x_1 = node_1.location[0]
                    x_2 = node_2.location[0]
                    w_1 = node_1.width
                    w_2 = node_2.width
                    cx_1 = x_1 + 0.5 * w_1
                    cx_2 = x_2 + 0.5 * w_2
                    rx_1 = 0.5 * w_1 + margin
                    rx_2 = 0.5 * w_2 + margin

                    # Note: "dimensions" and "height" may not be correct depending on the situation
                    def get_height(node: bpy.types.Node) -> float:
                        if node.dimensions.y > epsilon:
                            return node.dimensions.y
                        elif math.fabs(node.height - 100.0) > epsilon:
                            return node.height
                        else:
                            return 200.0

                    y_1 = node_1.location[1]
                    y_2 = node_2.location[1]
                    h_1 = get_height(node_1)
                    h_2 = get_height(node_2)
                    cy_1 = y_1 - 0.5 * h_1
                    cy_2 = y_2 - 0.5 * h_2
                    ry_1 = 0.5 * h_1 + margin
                    ry_2 = 0.5 * h_2 + margin

                    C_x = math.fabs(cx_1 - cx_2) - (rx_1 + rx_2)
                    C_y = math.fabs(cy_1 - cy_2) - (ry_1 + ry_2)

                    # If no collision, just skip
                    if C_x >= 0.0 or C_y >= 0.0:
                        continue

                    # Solve collision for the "easier" direction
                    if C_x > C_y:
                        grad_C_x_1 = 1.0 if cx_1 - cx_2 >= 0.0 else -1.0
                        grad_C_x_2 = -1.0 if cx_1 - cx_2 >= 0.0 else 1.0
                        lagrange = C_x / (grad_C_x_1 * grad_C_x_1 + grad_C_x_2 * grad_C_x_2)
                        delta_x_1 = -lagrange * grad_C_x_1
                        delta_x_2 = -lagrange * grad_C_x_2

                        node_1.location[0] += k * delta_x_1
                        node_2.location[0] += k * delta_x_2

                        squared_deltas_sum += k * k * (delta_x_1 * delta_x_1 + delta_x_2 * delta_x_2)
                    else:
                        grad_C_y_1 = 1.0 if cy_1 - cy_2 >= 0.0 else -1.0
                        grad_C_y_2 = -1.0 if cy_1 - cy_2 >= 0.0 else 1.0
                        lagrange = C_y / (grad_C_y_1 * grad_C_y_1 + grad_C_y_2 * grad_C_y_2)
                        delta_y_1 = -lagrange * grad_C_y_1
                        delta_y_2 = -lagrange * grad_C_y_2

                        node_1.location[1] += k * delta_y_1
                        node_2.location[1] += k * delta_y_2

                        squared_deltas_sum += k * k * (delta_y_1 * delta_y_1 + delta_y_2 * delta_y_2)

        if verbose:
            print("Iteration #" + str(i) + ": " + str(previous_squared_deltas_sum - squared_deltas_sum))

        # Check the termination condition
        if math.fabs(previous_squared_deltas_sum - squared_deltas_sum) < epsilon:
            if second_stage:
                break
            else:
                target_space = 0.5 * target_space
                second_stage = True

        previous_squared_deltas_sum = squared_deltas_sum


# =============================================================================
# NODE END
# =============================================================================

# =============================================================================
# TEXTURE
# =============================================================================

def add_clouds_texture(name: str = "Clouds Texture",
                       size: float = 0.25,
                       depth: int = 2,
                       nabla: float = 0.025,
                       brightness: float = 1.0,
                       contrast: float = 1.0) -> bpy.types.CloudsTexture:
    """
    https://docs.blender.org/api/current/bpy.types.BlendDataTextures.html
    https://docs.blender.org/api/current/bpy.types.Texture.html
    https://docs.blender.org/api/current/bpy.types.CloudsTexture.html
    """

    # TODO: Check whether the name is already used or not

    tex = bpy.data.textures.new(name, type='CLOUDS')

    tex.noise_scale = size
    tex.noise_depth = depth
    tex.nabla = nabla

    tex.intensity = brightness
    tex.contrast = contrast

    return tex


# =============================================================================
# TEXTURE END
# =============================================================================

# =============================================================================
# UTILS
# =============================================================================


# /////////////////////////////////////////////////////////////////////////////
# Text
# /////////////////////////////////////////////////////////////////////////////

def create_text(
        scene: bpy.types.Scene,
        body: str,
        name: str,
        align_x: str = 'CENTER',
        align_y: str = 'CENTER',
        size: float = 1.0,
        font_name: str = "Bfont",
        extrude: float = 0.0,
        space_line: float = 1.0,
        location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> bpy.types.Object:
    new_text_data: bpy.types.Curve = bpy.data.curves.new(name=name, type='FONT')

    new_text_data.body = body
    new_text_data.align_x = align_x
    new_text_data.align_y = align_y
    new_text_data.size = size
    new_text_data.font = bpy.data.fonts[font_name]
    new_text_data.space_line = space_line
    new_text_data.extrude = extrude

    new_object: bpy.types.Object = bpy.data.objects.new(name, new_text_data)
    scene.collection.objects.link(new_object)

    new_object.location = location
    new_object.rotation_euler = (math.pi * rotation[0] / 180.0, math.pi * rotation[1] / 180.0, math.pi * rotation[2])

    return new_object


# /////////////////////////////////////////////////////////////////////////////
# Text End
# /////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////
# Scene
# /////////////////////////////////////////////////////////////////////////////


def set_animation(scene: bpy.types.Scene,
                  fps: int = 24,
                  frame_start: int = 1,
                  frame_end: int = 48,
                  frame_current: int = 1) -> None:
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_current = frame_current


def build_rgb_background(world: bpy.types.World,
                         rgb: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
                         strength: float = 1.0) -> None:
    world.use_nodes = True
    node_tree = world.node_tree

    rgb_node = node_tree.nodes.new(type="ShaderNodeRGB")
    rgb_node.outputs["Color"].default_value = rgb

    node_tree.nodes["Background"].inputs["Strength"].default_value = strength

    node_tree.links.new(rgb_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

    arrange_nodes(node_tree)


def build_environment_texture_background(world: bpy.types.World, hdri_path: str, rotation: float = 0.0) -> None:
    world.use_nodes = True
    node_tree = world.node_tree

    environment_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    environment_texture_node.image = bpy.data.images.load(hdri_path)

    mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
    if bpy.app.version >= (2, 81, 0):
        mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, rotation)
    else:
        mapping_node.rotation[2] = rotation

    tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")

    node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    node_tree.links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
    node_tree.links.new(environment_texture_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

    arrange_nodes(node_tree)


def set_output_properties(scene: bpy.types.Scene,
                          resolution_percentage: int = 100,
                          output_file_path: str = "",
                          res_x: int = 1920,
                          res_y: int = 1080) -> None:
    scene.render.resolution_percentage = resolution_percentage
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    if output_file_path:
        scene.render.filepath = output_file_path


def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = False,
                        prefer_cuda_use: bool = True,
                        use_adaptive_sampling: bool = False) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.use_adaptive_sampling = use_adaptive_sampling
    scene.cycles.samples = num_samples

    # Enable GPU acceleration
    # Source - https://blender.stackexchange.com/a/196702
    if prefer_cuda_use:
        bpy.context.scene.cycles.device = "GPU"

        # Change the preference setting
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    # Call get_devices() to let Blender detects GPU device (if any)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Let Blender use all available devices, include GPU and CPU
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    # Display the devices to be used for rendering
    print("----")
    print("The following devices will be used for path tracing:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("- {}".format(d["name"]))
    print("----")


# /////////////////////////////////////////////////////////////////////////////
# Scene End
# /////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////
# Constraints
# /////////////////////////////////////////////////////////////////////////////


def add_track_to_constraint(camera_object: bpy.types.Object, track_to_target_object: bpy.types.Object) -> None:
    constraint = camera_object.constraints.new(type='TRACK_TO')
    constraint.target = track_to_target_object
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'


def add_copy_location_constraint(copy_to_object: bpy.types.Object,
                                 copy_from_object: bpy.types.Object,
                                 use_x: bool,
                                 use_y: bool,
                                 use_z: bool,
                                 bone_name: str = '') -> None:
    constraint = copy_to_object.constraints.new(type='COPY_LOCATION')
    constraint.target = copy_from_object
    constraint.use_x = use_x
    constraint.use_y = use_y
    constraint.use_z = use_z
    if bone_name:
        constraint.subtarget = bone_name


# /////////////////////////////////////////////////////////////////////////////
# Constraints End
# /////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////
# Library
# /////////////////////////////////////////////////////////////////////////////


def append_material(blend_file_path: str, material_name: str) -> bool:
    """
    https://docs.blender.org/api/current/bpy.types.BlendDataLibraries.html
    """

    # Load the library file
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        # Check whether the specified material exists in the blend file
        if material_name in data_from.materials:
            # Append the material and return True
            data_to.materials = [material_name]
            return True
        else:
            # If the material is not found, return False without doing anything
            return False

    # TODO: Handle the exception of not being able to load the library file
    # TODO: Remove the linked library from byp.data.libraries


# /////////////////////////////////////////////////////////////////////////////
# Library End
# /////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////
# Misc
# /////////////////////////////////////////////////////////////////////////////

def clean_objects() -> None:
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

# /////////////////////////////////////////////////////////////////////////////
# Misc End
# /////////////////////////////////////////////////////////////////////////////

# =============================================================================
# UTILS END
# =============================================================================
