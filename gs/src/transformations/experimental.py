import numpy as np
import pandas as pd
from typing import List


def rotation_matrix(theta: np.array) -> np.array:
    '''
    given a vector of three angles in radiant,
    return the corrisponding 3x3 rotation matrix
    '''
    theta_x, theta_y, theta_z = theta
    cos_theta_x = np.cos(theta_x)
    sin_theta_x = np.sin(theta_x)
    cos_theta_y = np.cos(theta_y)
    sin_theta_y = np.sin(theta_y)
    cos_theta_z = np.cos(theta_z)
    sin_theta_z = np.sin(theta_z)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_theta_x, -sin_theta_x],
        [0, sin_theta_x, cos_theta_x],
    ])
    R_y = np.array([
        [cos_theta_y, 0, sin_theta_y],
        [0, 1, 0],
        [-sin_theta_y, 0, cos_theta_y],
    ])
    R_z = np.array([
        [cos_theta_z, -sin_theta_z, 0],
        [sin_theta_z, cos_theta_z, 0],
        [0, 0, 1],
    ])

    return R_x.dot(R_y).dot(R_z)

def apply_rotation(df: pd.DataFrame, cols_xyz: List[str], rad_xyz: List[str]) -> pd.DataFrame:
    matrixes = (-df[rad_xyz]).apply(rotation_matrix, axis=1)
    vectors = df[cols_xyz].values

    new_coords = list(map(
        lambda M_v: M_v[0].dot(M_v[1]),
        zip(matrixes, vectors)
    ))
    df[[f'~{c}' for c in cols_xyz]] = np.stack(new_coords)
    return df