import numpy as np
import os
import sys
import math

class Construct_EXYZ:

    def __init__(self, xyz_file=None, force_file=None, output_file = None, stress_file = None):

        files = [file for file in os.listdir()]

        # If xyz_file, output_file or force_file are not passed, find them in the directory
        if output_file is None:
            for file in files:
                if '.out' in file:
                    output_file = file
                    break

        if force_file is None:
            for file in files:
                if '_0.xyz' in file:
                    force_file = file
                    break

        if xyz_file is None:
            for file in files:
                if 'min.xyz' in file:
                    xyz_file = file
                    break
        if stress_file is None:
            for file in files:
                if '.stress_tensor' in file:
                    stress_file = file
                    break

        # Error handling if files not found
        if force_file is None:
            print('ERROR: force file not found, ensure it contains the string "frc", or pass name manually (force_file =)')
            sys.exit()
        if xyz_file is None:
            print('ERROR: position file not found, ensure it contains the string "pos", or pass file name manually (xyz_file =)')
            sys.exit()
        if output_file is None:
            print('ERROR: cp2k output file not found, ensure it contains the string ".out", or pass name manually (output_file =)')
            sys.exit()
        if stress_file is None:
            print('ERROR: stress tensor file not found, ensure it contains the string ".stress_tensor", or pass name manually (stress_file =)')
            sys.exit()

        # Set class attributes
        self.xyz_file = xyz_file
        self.force_file = force_file
        self.output_file = output_file
        self.stress_file = stress_file

        # Extract number of atoms from xyz_file
        if 'xyz' in xyz_file:
            with open(self.xyz_file, 'r') as file:
                lines = file.readlines()

                self.n_atoms = int(lines[0].split()[0])





    def extract_xyz(self, input_file = None):


        if input_file == None:
            input_file = self.xyz_file
        atoms = []
        positions = []

        if '.xyz' in input_file:


            with open(input_file, 'r') as file:
                lines = file.readlines()
            for line in lines[2:]:
                elements = line.split()
                atom, x, y, z = str(elements[0]), float(elements[1]), float(elements[2]), float(elements[3])
                atoms.append(atom)
                positions.append((x, y, z))

            if ';' in lines[1]:

                lattice = lines[1].split(';')[0].split(':')[1].strip().split()
                lattice = [float(part.split('=')[1]) for part in lattice]



                angles = lines[1].split(';')[1].split(':')[1].split()[0:3]
                alpha, beta, gamma = [float(angle.split('=')[1]) for angle in angles]
            else:
                lattice = None
        elif '.inp' in input_file:

            with open(input_file, 'r') as file:
                lines = file.readlines()
            for index, line in enumerate(lines):
                if '&COORD' in line:
                    pos_index = index + 1
                if '&END COORD' in line:
                    pos_index_end = index
                if 'ABC' in line:
                    lattice_index = index
            pos_parts = lines[pos_index:pos_index_end]
            self.n_atoms = len(pos_parts)
            for line in pos_parts:
                elements = line.split()
                atom, x, y, z = str(elements[0]), float(elements[1]), float(elements[2]), float(elements[3])
                atoms.append(atom)
                positions.append((x, y, z))
            A, B, C = lines[lattice_index].split()[1:4]
            lattice = f'{A} 0 0 0 {B} 0 0 0 {C}'
        else:
            print('error retrieving xyz!')




        return atoms, positions, lattice, alpha, beta, gamma

    def count_atoms(self, atoms):
        Ba_count = 0
        O_count = 0
        Cu_count = 0
        Y_count = 0

        for atom in atoms:
            if atom == 'Ba':
                Ba_count += 1
            elif atom == 'O':
                O_count += 1
            elif atom == 'Cu':
                Cu_count += 1
            elif atom == 'Y':
                Y_count += 1
        return Ba_count, O_count, Cu_count, Y_count




    def extract_forces(self, input_file = None):
        forces = []
        if input_file == None:
            input_file = self.force_file
        with open(input_file, 'r') as file:
            lines = file.readlines()
        non_empty_lines = [line for line in lines if line.strip()]
        for line in non_empty_lines[2:(2+self.n_atoms)]:
            elements = line.split()
            x_frc, y_frc, z_frc = float(elements[3]) * 51.422, float(elements[4]) * 51.422, float(elements[5]) * 51.422
            forces.append((x_frc, y_frc, z_frc))


        return forces

    def extract_energy(self, input_file = None, Ba_count = None, Cu_count= None, O_count = None, Y_count = None ):

        binding_energy = {
            "Ba":-692.188,
            "O":-430.076,
            "Y":-1038.578,
            "Cu":-1305.806
        }


        if input_file == None:
            input_file = self.output_file
        energy = None
        with open (input_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if ' ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:' in line:
                energy = float(line.split()[8]) *27.2113
                break
        binding = (binding_energy["Ba"] * Ba_count) + (binding_energy["Cu"] * Cu_count) + (binding_energy["O"] * O_count) + (binding_energy["Y"] * Y_count)
        energy_w_binding_correction = energy - binding

        return energy, energy_w_binding_correction


    def lattice_vectors (self):

        _, _, lattice, alpha, beta, gamma = self.extract_xyz(input_file = self.xyz_file)
        a = lattice[0]
        b = lattice[1]
        c = lattice[2]

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        alpha = math.radians(alpha)
        beta = math.radians(beta)
        gamma = math.radians(gamma)
        lattice2 = [0] * 9
        lattice2[0] = a  # a1
        lattice2[1] = 0.0
        lattice2[2] = 0.0

        lattice2[3] = b * math.cos(gamma)  # b1
        lattice2[4] = b * math.sin(gamma)  # b2
        lattice2[5] = 0.0

        lattice2[6] = c * math.cos(beta)  # c1
        lattice2[7] = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)  # c2
        lattice2[8] = (a*b*c*math.sqrt(1-math.cos(alpha)**2-math.cos(beta)**2-math.cos(gamma)**2+2*math.cos(alpha)*math.cos(beta)*math.cos(gamma)))/(a*b*math.sin(gamma))  # c3


        lattice_line = " ".join(map(str, lattice2))
        # volume = np.abs(np.linalg.det(lattice_vectors))
        return lattice_line


    def extract_stress(self, input_file = None):

        def calculate_volume(a, b, c, alpha, beta, gamma):

            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            alpha = math.radians(alpha)
            beta = math.radians(beta)
            gamma = math.radians(gamma)

            # Calculate cosines of the angles
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            cos_gamma = math.cos(gamma)



            # Calculate the volume
            volume = a * b * c * math.sqrt(
                1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
            )

            return volume


        _, _, lattice, alpha, beta, gamma = self.extract_xyz(input_file = self.xyz_file)
        a = lattice[0]
        b = lattice[1]
        c = lattice[2]



        volume = calculate_volume(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)



        if input_file == None:
            input_file = self.stress_file

        with open (input_file, 'r') as f:
            lines = f.readlines()
        for index, line in enumerate(lines):
            if 'Analytical stress tensor [GPa]' in line:
                stress_lines = [index + 2, index +3, index+4]
                break
        stress_string = str()
        for line in lines[stress_lines[0]:stress_lines[2] + 1]:
            x, y, z = np.array([float(x) for x in line.split()[2:5]]) * volume * -0.006241509  #convert from Gpa to eV
            stress_string += f'{x} {y} {z} '



        return stress_string.rstrip()











    def get_pos_force_ener_stress(self):

        atoms, positions, lattice, _, _, _ = self.extract_xyz()
        lattice_line = self.lattice_vectors()
        Ba_count, O_count, Cu_count, Y_count = self.count_atoms(atoms)
        forces = self.extract_forces()
        energy, binding_corrected_energy = self.extract_energy(Ba_count = Ba_count, Cu_count = Cu_count, O_count = O_count, Y_count= Y_count)
        stress = self.extract_stress()

        return positions, forces, energy, atoms, lattice, stress, binding_corrected_energy, lattice_line

    def get_pos_force_ener(self):
        atoms, positions, lattice, _, _, _ = self.extract_xyz()
        lattice_line = self.lattice_vectors()
        Ba_count, O_count, Cu_count, Y_count = self.count_atoms(atoms)
        forces = self.extract_forces()
        energy, binding_corrected_energy = self.extract_energy(Ba_count = Ba_count, Cu_count = Cu_count, O_count = O_count, Y_count= Y_count)

        return positions, forces, energy, atoms, lattice, binding_corrected_energy, lattice_line



    def create_database(self, config_type=None, include_stress=True):
        data = []

        # Set default config_type if not specified
        if config_type is None:
            config_type = 'crystal'
            print('Note that config type is not specified! Defaulting to crystal.')

        config_type = str(config_type)
        # Retrieve data depending on whether stress is included
        if include_stress:
            positions, forces, energy, atoms, lattice, stress, binding_corrected_energy, lattice_line = self.get_pos_force_ener_stress()
            comment_string = (
                f'dft_energy={energy} Lattice="{lattice_line}" dft_virial="{stress}" '
                f'Properties=species:S:1:pos:R:3:dft_force:R:3 '
                f'binding_corrected_energy={binding_corrected_energy} config_type={config_type}'
            )
        else:
            positions, forces, energy, atoms, lattice, binding_corrected_energy, lattice_line = self.get_pos_force_ener()
            comment_string = (
                f'dft_energy={energy} Lattice="{lattice_line}" '
                f'Properties=species:S:1:pos:R:3:dft_force:R:3 '
                f'binding_corrected_energy={binding_corrected_energy} config_type={config_type}'
            )

        data.append(self.n_atoms)
        data.append(comment_string)

        # Append atom data to `data`
        for i in range(self.n_atoms):
            atom_data = f'{atoms[i]} {" ".join(map(str, positions[i]))} {" ".join(map(str, forces[i]))}'
            data.append(atom_data)

        return data






all_data = []

volume_list = [-5, 0, 5]
strain = [0, 1, 2]
jiggle = [0, 1, 2, 3]

for vol in volume_list:
    for i in strain:
        for j in jiggle:
            path = f'vol{vol}/strain{i}/jiggle{j}'
            exyz = Construct_EXYZ(xyz_file = f'{path}/supercell_min.xyz', force_file = f'{path}/filename-1_0.xyz', output_file = f'{path}/result.out', stress_file=f'{path}/filename-1_0.stress_tensor')
            data = exyz.create_database(config_type='crystal')
            print(f'{path} database created!')
            all_data.append(data)

with open('database.xyz', 'w') as f:
    for data in all_data:
        for line in data:
            f.write(f'{line}\n')
