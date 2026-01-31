import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a MJCF into USD format."
)
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--fix-base",
    action="store_true",
    default=False,
    help="Fix the base to where it is imported.",
)

parser.add_argument(
    "--import-sites",
    action="store_true",
    default=False,
    help="Import sites by parsing the <site> tag.",
)

parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=True,
    help="Make the asset instanceable for efficient cloning.",
)

parser.add_argument(
    "--flatten",
    action="store_true",
    default=True,
    help="Export a flattened USD stage into a single file.",
)

parser.add_argument(
    "--flatten-cleanup",
    action="store_true",
    default=True,
    help="Delete newly generated sub-USD files after flatten export.",
)

parser.add_argument(
    "--keep-worldbody",
    action="store_true",
    default=False,
    help="Keep the extra 'worldBody' prim if the converter adds it.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import carb  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import omni.kit.app  # noqa: E402
from pxr import Usd, UsdPhysics, Gf, Sdf  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def _ensure_mjcf_importer_available() -> None:
    try:
        app = omni.kit.app.get_app()
        ext_mgr = app.get_extension_manager()
        if ext_mgr is None:
            return

        candidate_extensions = (
            "omni.importer.mjcf",
            "omni.isaac.mjcf",
            "omni.isaac.mjcf_importer",
            "omni.kit.mjcf",
        )

        for ext_id in candidate_extensions:
            try:
                if hasattr(ext_mgr, "set_extension_enabled_immediate"):
                    ext_mgr.set_extension_enabled_immediate(ext_id, True)
            except Exception:
                pass

        try:
            for _ in range(5):
                app.update()
        except Exception:
            pass
    except Exception:
        return


_ensure_mjcf_importer_available()


def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    usd_dir = os.path.dirname(dest_path)
    pre_existing_usd_files: set[str] = set()
    if args_cli.flatten_cleanup:
        usd_dir_path = Path(usd_dir)
        if usd_dir_path.exists():
            for file_path in usd_dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                    pre_existing_usd_files.add(str(file_path.resolve()))

    # Ensure the USD directory exists
    os.makedirs(usd_dir, exist_ok=True)

    # create the converter configuration
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=usd_dir,
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        import_sites=args_cli.import_sites,
        force_usd_conversion=True,
        make_instanceable=args_cli.make_instanceable,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {mjcf_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create mjcf converter and import the file
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    # print output
    print("MJCF importer output:")
    print(f"Generated USD file: {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    def remove_worldbody(stage: Usd.Stage) -> bool:
        removed = False
        candidate_paths: set[str] = set()
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            for name in ("worldBody", "worldbody"):
                candidate_paths.add(str(default_prim.GetPath().AppendChild(name)))

        for path in ("/worldBody", "/worldbody"):
            candidate_paths.add(path)

        for prim in stage.Traverse():
            if prim and prim.IsValid() and prim.GetName().lower() == "worldbody":
                candidate_paths.add(str(prim.GetPath()))

        for path in sorted(candidate_paths, key=len, reverse=True):
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                stage.RemovePrim(path)
                removed = True

        return removed

    def fix_references(stage: Usd.Stage, base_dir: str) -> None:
        from pxr import Sdf
        
        def rewrite_asset_path(asset_path: str) -> str | None:
            if not asset_path:
                return asset_path
            if asset_path.endswith(".py"):
                return None
            if os.path.isabs(asset_path):
                return asset_path
            abs_path = os.path.abspath(os.path.join(base_dir, asset_path))
            if os.path.exists(abs_path):
                return abs_path
            return asset_path

        root_layer = stage.GetRootLayer()
        if root_layer is not None:
            updated_sublayers: list[str] = []
            for sub_layer_path in list(root_layer.subLayerPaths):
                if sub_layer_path and not os.path.isabs(sub_layer_path):
                    abs_path = os.path.join(base_dir, sub_layer_path)
                    if os.path.exists(abs_path):
                        updated_sublayers.append(abs_path)
                    else:
                        updated_sublayers.append(sub_layer_path)
                else:
                    updated_sublayers.append(sub_layer_path)
            root_layer.subLayerPaths = updated_sublayers

        list_attrs = [
            "addedItems",
            "appendedItems",
            "deletedItems",
            "explicitItems",
            "orderedItems",
            "prependedItems",
        ]

        for prim in stage.Traverse():
            for prim_spec in prim.GetPrimStack():
                for list_op_name, item_type in (
                    ("referenceList", Sdf.Reference),
                    ("payloadList", Sdf.Payload),
                ):
                    list_op = getattr(prim_spec, list_op_name, None)
                    if list_op is None:
                        continue

                    for attr_name in list_attrs:
                        proxy = getattr(list_op, attr_name, None)
                        if proxy is None:
                            continue

                        original_items = list(proxy)
                        updated_items = []
                        changed = False

                        for item in original_items:
                            asset_path = getattr(item, "assetPath", None)
                            if asset_path is None:
                                asset_path = item.GetAssetPath()

                            new_asset_path = rewrite_asset_path(asset_path)
                            if new_asset_path is None:
                                changed = True
                                continue

                            if new_asset_path == asset_path:
                                updated_items.append(item)
                                continue

                            prim_path = getattr(item, "primPath", None)
                            if prim_path is None:
                                prim_path = item.GetPrimPath()
                            layer_offset = getattr(item, "layerOffset", None)
                            if layer_offset is None:
                                layer_offset = item.GetLayerOffset()

                            updated_items.append(
                                item_type(
                                    assetPath=new_asset_path,
                                    primPath=prim_path,
                                    layerOffset=layer_offset,
                                )
                            )
                            changed = True
                            
                        if changed:
                            proxy[:] = updated_items

    def fix_mass_properties(stage: Usd.Stage, mjcf_path: str) -> None:
        """
        Explicitly set mass properties from MuJoCo model to USD prims.
        This ensures that even if the importer missed them (e.g. calculated from geom),
        they are present in the USD.
        """
        print(f"Fixing mass properties using MuJoCo model from: {mjcf_path}")

        def load_mujoco_model_for_mass_fix(xml_path: str) -> mujoco.MjModel | None:
            tmp_path = None
            try:
                root = ET.parse(xml_path).getroot()
                modified = False
                for geom in root.findall(".//geom"):
                    if geom.get("mass") is not None:
                        continue
                    contype = geom.get("contype")
                    conaffinity = geom.get("conaffinity")
                    geom_type = geom.get("type")
                    geom_name = geom.get("name") or ""
                    if contype == "0" and conaffinity == "0" and (geom_type == "mesh" or geom_name.endswith("_visual")):
                        geom.set("mass", "0")
                        modified = True

                if not modified:
                    return mujoco.MjModel.from_xml_path(xml_path)

                mjcf_dir = os.path.dirname(os.path.abspath(xml_path))
                with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=mjcf_dir) as f:
                    f.write(ET.tostring(root, encoding="unicode"))
                    tmp_path = f.name
                return mujoco.MjModel.from_xml_path(tmp_path)
            except Exception as e:
                print(f"Failed to load MuJoCo model for mass fixing: {e}")
                return None
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    with contextlib.suppress(OSError):
                        os.remove(tmp_path)

        try:
            m = load_mujoco_model_for_mass_fix(mjcf_path)
        except Exception:
            m = None

        if m is None:
            return

        # Helper to find prim in stage
        # We assume the importer preserves body names.
        # Since we don't know the exact USD path structure easily, we'll search by name.
        # This might be ambiguous if names are not unique, but MuJoCo requires unique names for bodies usually.
        
        # Build a map of name -> prim list (to handle duplicates or hierarchy)
        name_to_prims = {}
        all_prim_paths = []
        for prim in stage.Traverse():
            name = prim.GetName()
            path = prim.GetPath().pathString
            all_prim_paths.append(path)
            if name not in name_to_prims:
                name_to_prims[name] = []
            name_to_prims[name].append(prim)

        # Iterate over all bodies in MuJoCo model (skip world=0)
        for i in range(1, m.nbody):
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if not body_name:
                continue
            
            target_prims = []
            # 1. Exact name match
            if body_name in name_to_prims:
                target_prims = name_to_prims[body_name]
            
            # 2. If not found, try suffix match on all paths (slower but safer)
            if not target_prims:
                for path in all_prim_paths:
                    if path.endswith("/" + body_name):
                        target_prims.append(stage.GetPrimAtPath(path))

            if not target_prims:
                print(f"Warning: Could not find USD prim for MuJoCo body '{body_name}'")
                # Debug: print some available paths
                # print(f"Available paths sample: {all_prim_paths[:5]}")
                continue
            
            # Apply to all matching prims (usually just one, but handle instances/duplicates)
            for prim in target_prims:
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                # Get properties
                mass = m.body_mass[i]
                inertia = m.body_inertia[i] # diagonal inertia
                ipos = m.body_ipos[i]       # CoM position relative to body frame
                iquat = m.body_iquat[i]     # CoM orientation (w, x, y, z)
                
                # Ensure it has RigidBodyAPI if possible, or at least MassAPI
                # UsdPhysics.MassAPI.Apply(prim)
                
                # Check if we need to apply PhysicsRigidBodyAPI
                # if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                #     UsdPhysics.RigidBodyAPI.Apply(prim)

                # Apply MassAPI
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateMassAttr(float(mass))
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*ipos))
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
                
                # MuJoCo quat is (w, x, y, z)
                quat = Gf.Quatf(float(iquat[0]), float(iquat[1]), float(iquat[2]), float(iquat[3]))
                mass_api.CreatePrincipalAxesAttr(quat)
                
                print(f"Set mass props for '{body_name}' at '{prim.GetPath()}': mass={mass:.3f}")

    def remove_stiffness_and_damping_attributes(stage: Usd.Stage) -> None:
        """Remove any attribute containing 'stiffness' or 'damping' from ALL prims."""
        count = 0
        for prim in stage.Traverse():
            # Get all property names
            prop_names = prim.GetPropertyNames()
            for attr_name in prop_names:
                # Check if the attribute contains stiffness or damping (case insensitive)
                name_lower = attr_name.lower()
                if "stiffness" in name_lower or "damping" in name_lower:
                        prim.RemoveProperty(attr_name)
                        count += 1
                         
        if count > 0:
            print(f"Removed {count} attributes containing 'stiffness' or 'damping' from the stage.")

    def patch_usd_file(usd_file_path: str) -> None:
        stage = Usd.Stage.Open(usd_file_path)
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {usd_file_path}")

        fix_references(stage, os.path.dirname(usd_file_path))
        
        # Remove unsupported attributes
        remove_stiffness_and_damping_attributes(stage)

        # Apply mass fix
        fix_mass_properties(stage, args_cli.input)

        if not args_cli.keep_worldbody:
            with contextlib.suppress(Exception):
                stage.Load()
            remove_worldbody(stage)

        stage.GetRootLayer().Save()

    usd_files_to_patch: list[str] = [mjcf_converter.usd_path]
    configuration_dir = os.path.join(usd_dir, "configuration")
    if os.path.isdir(configuration_dir):
        for file_path in Path(configuration_dir).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                usd_files_to_patch.append(str(file_path))

    for file_path in sorted(set(usd_files_to_patch)):
        patch_usd_file(file_path)

    stage: Usd.Stage | None = None
    if args_cli.flatten:
        stage = Usd.Stage.Open(mjcf_converter.usd_path)
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {mjcf_converter.usd_path}")
        dest_path_obj = Path(dest_path)
        tmp_flatten_path = dest_path_obj.with_suffix(".flatten_tmp" + dest_path_obj.suffix)
        if tmp_flatten_path.exists():
            tmp_flatten_path.unlink()
        stage.Export(str(tmp_flatten_path))
        os.replace(str(tmp_flatten_path), dest_path)

        print(f"Flattened USD file: {dest_path}")
        print("-" * 80)
        print("-" * 80)

        if args_cli.flatten_cleanup:
            post_existing_usd_files: set[str] = set()
            usd_dir_path = Path(usd_dir)
            if usd_dir_path.exists():
                for file_path in usd_dir_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                        post_existing_usd_files.add(str(file_path.resolve()))

            flattened_abs = str(Path(dest_path).resolve())
            newly_generated = post_existing_usd_files - pre_existing_usd_files
            for file_path in sorted(newly_generated):
                if file_path == flattened_abs:
                    continue
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        
        # Re-open the flattened stage and apply mass fix again to be absolutely sure
        print(f"Applying mass fix to flattened stage: {dest_path}")
        flat_stage = Usd.Stage.Open(dest_path)
        if flat_stage:
            remove_stiffness_and_damping_attributes(flat_stage)
            fix_mass_properties(flat_stage, args_cli.input)
            flat_stage.GetRootLayer().Save()
        else:
            print(f"Failed to open flattened stage for final fix: {dest_path}")

    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(dest_path if args_cli.flatten else mjcf_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
