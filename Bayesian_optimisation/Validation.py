from ase.io import read
import numpy as np


class Validation:
    def compute_mae(self, forces_ml, forces_ref):

        # Ensure the arrays have the same shape
        if forces_ml.shape != forces_ref.shape:
            raise ValueError("The shapes of forces_ml and forces_ref must match.")

        # Compute the absolute difference between ML and reference forces
        abs_diff = np.abs(forces_ml - forces_ref)

        # Compute the MAE
        mae = np.mean(abs_diff)

        return mae


    def compute_rmse(self, forces_ml, forces_ref):

        # Ensure the arrays have the same shape
        if forces_ml.shape != forces_ref.shape:
            raise ValueError("The shapes of forces_ml and forces_ref must match.")

        # Compute the squared difference between ML and reference forces
        squared_diff = np.square(forces_ml - forces_ref)

        # Compute the RMSE
        rmse = np.sqrt(np.mean(squared_diff))

        return rmse


    def get_force_ener (self, input_file = 'quip_test.xyz'):

        atoms_list = read(input_file, index=":")
        all_force_mae = []
        all_force_rmse = []
        all_energy_mae = []
        all_energy_rmse = []
        dft_energies = []
        ml_energies = []
        dft_forces = []
        ml_forces = []
        # Loop through each configuration
        for i, atoms in enumerate(atoms_list):

            num_atoms = len(atoms)
            # Perhaps make this flexible for different names
            if 'dft_force' in atoms.arrays:
                dft_force = atoms.arrays['dft_force']
            if 'force' in atoms.arrays:
                ml_force = atoms.arrays['force']
            if 'dft_energy' in atoms.info:
                dft_energy = atoms.info.get('dft_energy') / num_atoms
            ml_energy = atoms.get_total_energy() / num_atoms

            dft_energies.append(dft_energy)
            dft_forces.append(dft_force)
            ml_energies.append(ml_energy)
            ml_forces.append(ml_force)



            lines = []
            for x,y in zip(ml_forces, dft_forces):
                for individual_force_ml, individual_force_dft in zip(x,y):
                    lines.append(f'{individual_force_ml[0]} {individual_force_dft[0]}\n')
                    lines.append(f'{individual_force_ml[1]} {individual_force_dft[1]}\n')
                    lines.append(f'{individual_force_ml[2]} {individual_force_dft[2]}\n')

            with open ('forces.txt', 'w') as f:
                f.writelines(lines)


            with open ('energies.txt', 'w') as f:

                for x,y in zip(ml_energies, dft_energies):
                    line = f'{x} {y}'
                    f.write(f'{line}\n')

            all_force_mae.append(self.compute_mae(forces_ml=ml_force, forces_ref=dft_force))
            all_force_rmse.append(self.compute_rmse(forces_ml=ml_force, forces_ref=dft_force))
            all_energy_mae.append(self.compute_mae(forces_ml=ml_energy, forces_ref=dft_energy))
            all_energy_rmse.append(self.compute_rmse(forces_ml=ml_energy, forces_ref=dft_energy))



        return np.mean(all_energy_rmse), np.mean(all_energy_mae), np.mean(all_force_rmse), np.mean(all_force_mae)
