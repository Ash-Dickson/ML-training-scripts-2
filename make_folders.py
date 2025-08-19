print('Loading libraries...')


'''STEPS FOR RUNNING FILE:

            - Create directory for system of interest
            - Run getxyz.py to make folders required for runs
            - ensure that in the directory above the working directory energy.inp and minmise.inp are included
            - Change the job script template for each system, and change BASIS SET and GTH_POTENTIALS file location for each system
            - Run script and follow instructions'''

import numpy as np
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import PerturbStructureTransformation, DeformStructureTransformation
from pymatgen.io.xyz import XYZ
import os
import shutil
import sys
import time
import subprocess
import random

print('Libraries loaded!')

def vol_scale(file_name, scale_factor, out_file_name):
    def read_structure(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        num_atoms = int(lines[0].strip())
        cell_info = lines[1].strip().split(';')

        # Extract lattice parameters
        lattice_params = cell_info[0].split(':')[1].strip().split()
        a = float(lattice_params[0].split('=')[1])
        b = float(lattice_params[1].split('=')[1])
        c = float(lattice_params[2].split('=')[1])

        # Extract angles
        angles = cell_info[1].split(':')[1].strip().split()
        A = float(angles[0].split('=')[1])
        B = float(angles[1].split('=')[1])
        C = float(angles[2].split('=')[1])

        atoms = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:  # Ensure the line has at least 4 parts (element + 3 coordinates)
                try:
                    element = parts[0]
                    x, y, z = map(float, parts[1:4])  # Convert the last three parts to floats
                    atoms.append((element, x, y, z))
                except ValueError:
                    print(f"Warning: Skipping malformed line in {file_name}: {line.strip()}")
                    continue

        return num_atoms, a, b, c, A, B, C, atoms


    def write_structure(file_name, num_atoms, a, b, c, atoms_new, A, B, C):
        with open(file_name, 'w') as file:
            # Write the number of atoms
            file.write(f"{num_atoms}\n")
            # Write the lattice parameters and angles as the second line (header)
            file.write(f"Lattice: a={a:.6f} b={b:.6f} c={c:.6f}; Angles: alpha={A:.6f} beta={B:.6f} gamma={C:.6f}\n")
            # Write the updated atomic positions
            for atom in atoms_new:
                element, x, y, z = atom
                file.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")

    def scale_structure(a, b, c, atoms, scale_factor):
        # Scale the lattice parameters by the volume scale factor (cubed root of the scale factor)
        volume_scale_factor = float(scale_factor) ** (1 / 3)
        a_new = a * volume_scale_factor
        b_new = b * volume_scale_factor
        c_new = c * volume_scale_factor

        # Scale the atomic positions
        atoms_new = []
        for atom in atoms:
            element, x, y, z = atom
            x_new = x * volume_scale_factor
            y_new = y * volume_scale_factor
            z_new = z * volume_scale_factor
            atoms_new.append((element, x_new, y_new, z_new))

        return a_new, b_new, c_new, atoms_new, volume_scale_factor

    # Read the structure
    num_atoms, a, b, c, A, B, C, atoms = read_structure(file_name)

    # Scale the structure
    a_new, b_new, c_new, atoms_new, volume_scale_factor = scale_structure(a, b, c, atoms, scale_factor)

    # Write the scaled structure to the output
    write_structure(out_file_name, num_atoms, a_new, b_new, c_new, atoms_new, A, B, C)

    return out_file_name

def replace_line(filename, string_to_replace, new_string):
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            for line in lines:
                if string_to_replace in line:
                    line = line.replace(string_to_replace, new_string)
                    f.write(line)
                else:
                    f.write(line)

def read_xyz(xyz_file):
        coords = []
        atoms = []
        with open(xyz_file, 'r') as inp:
            lines = inp.readlines()
            comment = lines[1].split(';')[0].split(':')[1].split()
            a = float(comment[0].split('=')[1])
            b = float(comment[1].split('=')[1])
            c = float(comment[2].split('=')[1])

            for line in lines[2:]:
                if line.strip():  # Ensure non-empty line
                    atom, x, y, z = line.split()[:4]
                    atoms.append(atom)
                    coords.append([float(x), float(y), float(z)])

        coords = np.array(coords)
        atoms = np.array(atoms)
        return coords, atoms, a, b, c


def lattice_params (structure):
    # Lattice parameters
    a = float(structure.lattice.matrix[0][0])
    b = float(structure.lattice.matrix[1][1])
    c = float(structure.lattice.matrix[2][2])

    lattice_matrix = structure.lattice.matrix
    a_vec = lattice_matrix[0]
    b_vec = lattice_matrix[1]
    c_vec = lattice_matrix[2]

    # Calculate lattice angles (in degrees)
    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (np.linalg.norm(b_vec) * np.linalg.norm(c_vec))))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(c_vec))))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec))))

    return a, b, c, alpha, beta, gamma

def apply_random_strain(file_name, out_file_name, edge):
    def read_structure(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        num_atoms = int(lines[0].strip())
        cell_info = lines[1].strip().split(';')

        # Extract lattice parameters
        lattice_params = cell_info[0].split(':')[1].strip().split()
        a = float(lattice_params[0].split('=')[1])
        b = float(lattice_params[1].split('=')[1])
        c = float(lattice_params[2].split('=')[1])

        # Extract original angles
        angles = cell_info[1].split(':')[1].strip().split()
        alpha = float(angles[0].split('=')[1])
        beta = float(angles[1].split('=')[1])
        gamma = float(angles[2].split('=')[1])

        # Extract atomic positions
        atoms = []
        for line in lines[2:]:
            parts = line.split()
            element = parts[0]
            x, y, z = map(float, parts[1:])
            atoms.append((element, x, y, z))

        return num_atoms, a, b, c, alpha, beta, gamma, atoms

    def write_structure(file_name, num_atoms, a_strain, b_strain, c_strain, atoms_new, alpha_strain, beta_strain, gamma_strain):
        with open(file_name, 'w') as file:
            file.write(f"{num_atoms}\n")
            file.write(f"Lattice: a={a_strain:.6f} b={b_strain:.6f} c={c_strain:.6f}; Angles: alpha={alpha_strain:.6f} beta={beta_strain:.6f} gamma={gamma_strain:.6f}\n")
            for atom in atoms_new:
                element, x, y, z = atom
                file.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")

    def apply_strain(a, b, c, atoms, edge, alpha, beta, gamma):
        # Apply random displacements to lattice parameters `a`, `b`, and `c`
        displacements = [np.random.uniform(-0.01 * edge, 0.01 * edge) for _ in range(3)]
        a_strain = a + displacements[0]
        b_strain = b + displacements[1]
        c_strain = c + displacements[2]

        # Gaussian adjustments around original angles with standard deviation to stay within ±5 degrees
        std_dev = 5 / 3  # Roughly 99.7% of values will stay within ±5 degrees
        alpha_strain = np.random.normal(alpha, std_dev)
        beta_strain = np.random.normal(beta, std_dev)
        gamma_strain = np.random.normal(gamma, std_dev)

        # Adjust atomic positions according to strained lattice parameters
        atoms_new = []
        for element, x, y, z in atoms:
            # Scale each atomic coordinate by the ratio of new to original lattice parameters
            x_new = x * (a_strain / a)
            y_new = y * (b_strain / b)
            z_new = z * (c_strain / c)
            atoms_new.append((element, x_new, y_new, z_new))

        return a_strain, b_strain, c_strain, atoms_new, alpha_strain, beta_strain, gamma_strain

    # Read structure and apply strain to lattice parameters and angles
    num_atoms, a, b, c, alpha, beta, gamma, atoms = read_structure(file_name)
    a_strain, b_strain, c_strain, atoms_new, alpha_strain, beta_strain, gamma_strain = apply_strain(a, b, c, atoms, edge, alpha, beta, gamma)

    # Write the strained structure to a new file
    write_structure(out_file_name, num_atoms, a_strain, b_strain, c_strain, atoms_new, alpha_strain, beta_strain, gamma_strain)

    return out_file_name

def jiggle(input_file, output_file, unit_cell_file):
    def read_xyz(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        num_atoms = int(lines[0].strip())
        cell_info = lines[1].strip().split(';')

        # Extract lattice parameters and angles
        lattice_params = cell_info[0].split(':')[1].strip().split()
        a = float(lattice_params[0].split('=')[1])
        b = float(lattice_params[1].split('=')[1])
        c = float(lattice_params[2].split('=')[1])

        angles = cell_info[1].split(':')[1].strip().split()
        alpha = float(angles[0].split('=')[1])
        beta = float(angles[1].split('=')[1])
        gamma = float(angles[2].split('=')[1])

        # Atomic positions
        atoms = []
        coords = []
        for line in lines[2:]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return coords, atoms, a, b, c, alpha, beta, gamma

    # Read lattice parameters from the strained supercell (input_file)
    coords, atoms, a, b, c, alpha, beta, gamma = read_xyz(input_file)

    # Read unit cell data to calculate max displacement
    #_, _, unit_a, unit_b, unit_c, _, _, _ = read_xyz(unit_cell_file)
    max_displacement = 0.1 * min(3.827, 3.869, 11.699)

    # Apply perturbation to coordinates
    perturbed_coords = []
    for coord in coords:
        perturbed_coord = [
            coord[0] + random.uniform(-max_displacement, max_displacement),
            coord[1] + random.uniform(-max_displacement, max_displacement),
            coord[2] + random.uniform(-max_displacement, max_displacement),
        ]
        perturbed_coords.append(perturbed_coord)

    # Write the perturbed coordinates to the output file
    with open(output_file, "w") as f:
        f.write(f"{len(atoms)}\n")
        comment = f"Lattice: a={a:.6f} b={b:.6f} c={c:.6f}; Angles: alpha={alpha:.6f} beta={beta:.6f} gamma={gamma:.6f}"
        f.write(f"{comment}\n")
        for atom, coord in zip(atoms, perturbed_coords):
            f.write(f"{atom} {' '.join(map(str, coord))}\n")

    print(f'Jiggle applied to {input_file}. Max displacement = {max_displacement:.6f}')


def create_folders(vol_list, num_strains, num_jiggles, supercell_file, unit_cell_file):
    for N in vol_list:
        vol_dir = f'vol{N}'
        if not os.path.exists(vol_dir):
            os.mkdir(vol_dir)

        # Apply volume scaling
        scaled_supercell_file = f'{vol_dir}/supercell_min.xyz'
        vol_scale(supercell_file, scale_factor=(1 + N / 100), out_file_name=scaled_supercell_file)
        _, _, edge = read_xyz(scaled_supercell_file)

        for strain_num in range(num_strains):
            strain_dir = f'{vol_dir}/strain{strain_num}'
            if not os.path.exists(strain_dir):
                os.mkdir(strain_dir)

            # Ensure that one strain is zero and copy (scaled) supercell_min
            if strain_num == 0:  # Zero strain
                shutil.copy(scaled_supercell_file, f'{strain_dir}/supercell_min.xyz')
                strained_supercell_file = f'{strain_dir}/supercell_min.xyz'
            else:
                # Apply random strain to non-zero cases
                strained_supercell_file = f'{strain_dir}/supercell_min.xyz'
                apply_random_strain(scaled_supercell_file, strained_supercell_file, edge)

            for jiggle_num in range(num_jiggles):
                jiggle_dir = f'{strain_dir}/jiggle{jiggle_num}'
                if not os.path.exists(jiggle_dir):
                    os.mkdir(jiggle_dir)

                if jiggle_num == 0:  # No jiggle
                    # Copy the strained supercell directly into jiggle0 folder
                    shutil.copy(strained_supercell_file, f'{jiggle_dir}/supercell_min.xyz')
                else:
                    # Apply jiggle to the strained supercell for non-zero cases
                    jiggle(strained_supercell_file, f'{jiggle_dir}/supercell_min.xyz', unit_cell_file)

                # Copy energy.inp to jiggle directory
                shutil.copy('energy.inp', jiggle_dir)

                # Update ABC and angles in energy.inp for each jiggle folder
                extract_comment_change_inp(input_file=f'{jiggle_dir}/energy.inp', xyz_file=f'{jiggle_dir}/supercell_min.xyz')

    print("Folder creation and file manipulation completed.")


def supercell (input_file, x, y, z):
    coords, atoms, a, b, c = read_xyz(input_file)

    lattice = [[a,0,0], [0,b,0], [0,0,c]]
    # Create structure object
    structure = Structure(lattice, atoms, coords, coords_are_cartesian = True)
    supercell = structure.make_supercell([[x, 0, 0], [0, y, 0], [0, 0, z]])
    a, b, c, alpha, beta, gamma = lattice_params(supercell)
    xyz = XYZ(supercell)
    xyz_string = xyz.__str__()
    num_atoms = int(xyz_string.splitlines()[0])

    # Add lattice parameters and angles as comment
    comment = f"Lattice: a={a:.6f} b={b:.6f} c={c:.6f}; Angles: alpha={alpha:.6f} beta={beta:.6f} gamma={gamma:.6f}"
    xyz_lines = xyz_string.split('\n')
    xyz_lines.pop(1) #Remove original comment line
    xyz_lines.insert(1, comment)
    modified_xyz_string = '\n'.join(xyz_lines)
    return modified_xyz_string, num_atoms

def create_supercell_min():
    try:
        # Read the entire file
        with open('supercell_min.xyz', 'r') as f:
            min_structure = f.readlines()

        # Extract comment line containing lattice parameters
        comment_line = min_structure[1]
        print('Extracted cell parameters from minimization!')

        # Extract lattice parameters a, b, c from the comment line
        # Assuming format "Lattice: a=... b=... c=..."
        import re
        match = re.search(r'a=([\d.]+)\s+b=([\d.]+)\s+c=([\d.]+)', comment_line)
        if not match:
            raise ValueError("Lattice parameters not found in comment line.")

        # Convert extracted values to floats
        a, b, c = map(float, match.groups())

        # Update the comment line with extracted parameters
        Lattice_comment = comment_line.split(';')[0]
        newline = comment_line.replace(Lattice_comment, f"Lattice: a={a:.6f} b={b:.6f} c={c:.6f}")
        min_structure[1] = newline

        # Write updated content back to file
        with open('supercell_min.xyz', 'w') as f:
            f.writelines(min_structure)

        print('supercell_min.xyz created with updated lattice parameters!')

        # Return edge (for strain calculations)
        edge = max(a, b, c)
        return edge

    except FileNotFoundError:
        print("Error: File 'supercell_min.xyz' not found.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None



def create_folders(vol_list, num_strains, num_jiggles, supercell_file, unit_cell_file):
    for N in vol_list:
        vol_dir = f'vol{N}'
        if not os.path.exists(vol_dir):
            os.mkdir(vol_dir)

        # Apply volume scaling
        scaled_supercell_file = f'{vol_dir}/supercell_min.xyz'
        vol_scale(supercell_file, scale_factor=(1 + N / 100), out_file_name=scaled_supercell_file)

        for strain_num in range(num_strains):
            strain_dir = f'{vol_dir}/strain{strain_num}'

            if not os.path.exists(strain_dir):
                os.mkdir(strain_dir)

            # Ensure that one strain is zero and copy (scaled) supercell_min
            if strain_num == 0:  # Zero strain
                shutil.copy(scaled_supercell_file, f'{strain_dir}/supercell_min.xyz')
                strained_supercell_file = f'{strain_dir}/supercell_min.xyz'
            else:
                # Apply random strain to non-zero cases
                strained_supercell_file = f'{strain_dir}/supercell_min.xyz'
                apply_random_strain(scaled_supercell_file, strained_supercell_file, edge)

            for jiggle_num in range(num_jiggles):
                jiggle_dir = f'{strain_dir}/jiggle{jiggle_num}'
                if not os.path.exists(jiggle_dir):
                    os.mkdir(jiggle_dir)

                # Ensure that one jiggle is zero and copy (strained) supercell_min
                if jiggle_num == 0:  # No jiggle
                    # Copy the strained supercell into jiggle0 folder
                    shutil.copy(strained_supercell_file, f'{jiggle_dir}/supercell_min.xyz')
                else:
                    # Apply jiggle to the strained supercell for non-zero cases
                    jiggle(strained_supercell_file, f'{jiggle_dir}/supercell_min.xyz', unit_cell_file='inp.xyz')

                # Copy energy.inp to jiggle directory
                shutil.copy('energy.inp', jiggle_dir)

                # Update ABC and angles in energy.inp for each jiggle folder
                extract_comment_change_inp(input_file=f'{jiggle_dir}/energy.inp', xyz_file=f'{jiggle_dir}/supercell_min.xyz')

    print("Folder creation and file manipulation completed.")

def extract_comment_change_inp (input_file, xyz_file, symmetry = None):
    with open(xyz_file, 'r') as f:
        comment = f.readlines()[1]
        x, y, z = comment.split(';')[0].split(':')[1].split()[0:3]
        ABC_string = f"{x.split('=')[1]} {y.split('=')[1]} {z.split('=')[1].replace(';', '')}"
        a, b, c = comment.split(';')[1].split(':')[1].split()[0:3]
        abc_string = f"{a.split('=')[1]} {b.split('=')[1]} {c.split('=')[1].replace(';', '')}"

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if '$ABC' in line:
                lines[index] = line.replace('$ABC', ABC_string)
            if '$abc' in line:
                lines[index] = line.replace('$abc', abc_string)

    with open(input_file, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":

    parent = os.getcwd()

############################################################ DEFORM MINIMISED CELL, MAKE RELEVENT DIRECTORIES AND WRITE PROGRESS TRACKER #################################
    edge = create_supercell_min()

    # Parameters for deformation
    vol_list = [-5, 0, 5]  # Volume percentage changes
    num_strains = 3  # Number of strains to apply per volume
    num_jiggles = 4  # Number of jiggles to apply per strain

    create_folders(vol_list, num_strains, num_jiggles, 'supercell_min.xyz', 'inp.xyz')
    print("Process completed successfully.")
