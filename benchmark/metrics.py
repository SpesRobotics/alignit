"""Metrics utilities for benchmark evaluation."""

import numpy as np


def compute_translation_error(pose_actual, pose_target):
    """
    Compute translation error in mm.
    
    Args:
        pose_actual: 4x4 transformation matrix (actual pose)
        pose_target: 4x4 transformation matrix (target pose)
    
    Returns:
        Translation error in mm
    """
    t_actual = pose_actual[:3, 3]
    t_target = pose_target[:3, 3]
    error_m = np.linalg.norm(t_actual - t_target)
    return error_m * 1000  # Convert to mm


def compute_rotation_error(pose_actual, pose_target):
    """
    Compute rotation error in degrees using angle-axis representation.
    
    Args:
        pose_actual: 4x4 transformation matrix (actual pose)
        pose_target: 4x4 transformation matrix (target pose)
    
    Returns:
        Rotation error in degrees
    """
    R_actual = pose_actual[:3, :3]
    R_target = pose_target[:3, :3]
    
    # Compute relative rotation
    R_rel = R_actual @ R_target.T
    
    # Convert to angle-axis and extract angle
    rot = R.from_matrix(R_rel)
    angle_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_pose_error(pose_actual, pose_target, trans_weight=1.0, rot_weight=1.0):
    """
    Compute combined pose error.
    
    Args:
        pose_actual: 4x4 transformation matrix
        pose_target: 4x4 transformation matrix
        trans_weight: Weight for translation error (mm)
        rot_weight: Weight for rotation error (degrees)
    
    Returns:
        dict with 'translation', 'rotation', and 'combined' errors
    """
    trans_err = compute_translation_error(pose_actual, pose_target)
    rot_err = compute_rotation_error(pose_actual, pose_target)
    
    # Normalize and combine
    combined = trans_weight * trans_err + rot_weight * rot_err
    
    return {
        "translation_mm": trans_err,
        "rotation_deg": rot_err,
        "combined": combined,
    }


def check_convergence(error, trans_tol_mm=5.0, rot_tol_deg=5.0):
    """
    Check if pose error is within acceptable tolerance.
    
    Args:
        error: dict from compute_pose_error()
        trans_tol_mm: Translation tolerance in mm
        rot_tol_deg: Rotation tolerance in degrees
    
    Returns:
        bool: True if converged
    """
    return (error["translation_mm"] <= trans_tol_mm and 
            error["rotation_deg"] <= rot_tol_deg)


def compute_statistics(errors):
    """
    Compute statistics over a list of errors.
    
    Args:
        errors: list of error dicts from compute_pose_error()
    
    Returns:
        dict with mean, std, min, max for each error type
    """
    trans_errors = [e["translation_mm"] for e in errors]
    rot_errors = [e["rotation_deg"] for e in errors]
    
    return {
        "translation": {
            "mean_mm": np.mean(trans_errors),
            "std_mm": np.std(trans_errors),
            "min_mm": np.min(trans_errors),
            "max_mm": np.max(trans_errors),
        },
        "rotation": {
            "mean_deg": np.mean(rot_errors),
            "std_deg": np.std(rot_errors),
            "min_deg": np.min(rot_errors),
            "max_deg": np.max(rot_errors),
        },
    }
