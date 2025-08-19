print('Loading dependencies...')
from Validation import *
from File_constructor import *
import os
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import LogExpectedImprovement
print('Dependencies loaded!')

class Bayesian_Optimisation:

    def __init__ (self, quip_path = '/work/e05/e05/ash141/codes/QUIP2/QUIP/build/archer2_mpich+openmp',
                  validation_database_path = 'YBCO_test.xyz', initial_samples = 10, _2b = True, _3b = True):

        self.quip_path = quip_path
        self.validation_database_path = validation_database_path
        self.pot_name = 'gp1.xml'
        self.initial_samples = initial_samples

        self._2b = _2b
        self._3b = _3b

        self.nsparse_2b = 20
        self.nsparse_3b = 200

        with open('train.sh', 'r') as f:
            lines = f.readlines()

        if self._2b:
            exec_2b = '$exec atoms_filename=$infile gap={$k2b_params} $settings gp_file=$outfile'
            lines[-1] = exec_2b


        if self._3b:
            exec_3b = '''$exec atoms_filename=$infile gap={{$k2b_Ba_Ba}:{$k2b_Cu_Cu}:{$k2b_O_O}:{$k2b_Y_Y}:{$k2b_Ba_Y}:{$k2b_Cu_Ba}:{$k2b_O_Ba}:{$k2b_O_Cu}:{$k2b_Cu_Y}:{$k2b_O_Y}:{$k3b_params}} $settings gp_file=$outfile'''
            lines[-1] = exec_3b
        with open('train.sh', 'w') as f:
            f.writelines(lines)



    def validate_potential(self):

        print('Running QUIP to validate potential...')
        # openmp compilation of QUIP to run standard valdiation simulations (angle_3b not compatible with mpi)
        openmp = '/work/e05/e05/ash141/codes/QUIP2/QUIP/build/archer2_openmp/quip'
        os.system(f"{openmp} E=T F=T atoms_filename={self.validation_database_path} param_filename={self.pot_name} | grep AT | sed 's/AT//' > quip_test.xyz")
        print('QUIP validation complete!')
        print('Calculaing errors...')
        self.energy_rmse, self.energy_mae, self.force_rmse , self.force_mae = Validation().get_force_ener()
        print('Errors calculated!')
        print(f'Energy RMSE={self.energy_rmse}\nForce RMSE={self.force_rmse}\nEnergy MAE={self.energy_mae}\nForce MAE={self.force_mae}')

    def update_param_file (self):

        self.file_object = File_constructor(ermse=self.energy_rmse, frmse=self.force_rmse,
                             emae=self.energy_mae, fmae=self.force_mae, _2b = self._2b, _3b = self._3b)
        if not os.path.exists('params.txt'):
            self.file_object.write_header()

        self.file_object.write_new_data()


    def check_convergence(self):

        self.converged = False
        with open('params.txt', 'r') as f:
            data = f.readlines()[2:]
        forces = []
        for dat in data:
            force_mae = dat.split()[-1]
            forces.append(float(force_mae))

        threshold = 1e-3

        # Check if the last three iterations have minimal change
        if len(forces) >= 3:  # Ensure there are at least 3 elements to compare
            changes = np.abs(np.diff(forces[-3:]))
            self.converged = np.all(changes < threshold)
        else:
            self.converged = False  # Not enough iterations to determine convergence




    def read_params(self):

        with open ('params.txt', 'r') as f:
            lines = f.readlines()
            self.num_evaluations = len(lines) - 2
            data = lines[2:]


        self.param_data = {
            **({'cutoff2b': [], 'delta2b': [], 'theta2b': [], 'sparse2b': []} if self._2b else {}),
            **({'cutoff3b': [], 'delta3b': [], 'theta3b': [], 'sparse3b': []} if self._3b else {}),
            'ermse': [],
            'frmse': [],
            'emae': [],
            'fmae': []
        }


        for iteration in data:


            elements = iteration.split()
            if self._2b and not self._3b:

                _, cutoff, delta, theta, sparse2b, ermse, frmse, emae, fmae = elements
                self.param_data['cutoff2b'].append(float(cutoff))
                self.param_data['delta2b'].append(float(delta))
                self.param_data['theta2b'].append(float(theta))
                self.param_data['sparse2b'].append(float(sparse2b))


            elif self._2b and self._3b:

                _, cutoff2b, delta2b, theta2b, sparse2b, cutoff3b, delta3b, theta3b, sparse3b, ermse, frmse, emae, fmae = elements
                self.param_data['cutoff2b'].append(float(cutoff2b))
                self.param_data['delta2b'].append(float(delta2b))
                self.param_data['theta2b'].append(float(theta2b))
                self.param_data['sparse2b'].append(float(sparse2b))

                self.param_data['cutoff3b'].append(float(cutoff3b))
                self.param_data['delta3b'].append(float(delta3b))
                self.param_data['theta3b'].append(float(theta3b))
                self.param_data['sparse3b'].append(float(sparse3b))

            else:
                # Invalid configuration
                raise ValueError("Cannot have only 3-body descriptors or no descriptors selected!")

            self.param_data['ermse'].append(float(ermse))
            self.param_data['frmse'].append(float(frmse))
            self.param_data['emae'].append(float(emae))
            self.param_data['fmae'].append(float(fmae))





    def generate_descriptor_string(self, descriptor_type, params):
            descriptor_value = 'distance_2b ' if descriptor_type == 'k2b_params' else 'angle_3b '
            return f'{descriptor_type}="' + descriptor_value + " ".join(f"{key}={value}" for key, value in params.items()) + '"'

    def update_train_script (self, cutoff, delta, theta, sparse2b, cutoff3b, delta3b, theta3b, sparse3b):

        k2b_dict, k3b_dict = self.file_object.get_params()


        k2b_dict.update({'cutoff': cutoff, 'delta': delta, 'theta_uniform': theta, 'n_sparse': sparse2b})
        k3b_dict.update({'cutoff': cutoff3b, 'delta': delta3b, 'theta_uniform': theta3b, 'n_sparse': sparse3b})


        with open('train.sh', 'r') as f:
            lines = f.readlines()

        if self._2b:
            for index, line in enumerate(lines):
                if 'k2b_cutoff=' in line:
                    lines[index] = f'k2b_cutoff="{k2b_dict["cutoff"]}"\n'
                if 'k2b_delta=' in line:
                    lines[index] = f'k2b_delta="{k2b_dict["delta"]}"\n'
                if 'k2b_theta=' in line:
                    lines[index] = f'k2b_theta="{k2b_dict["theta_uniform"]}"\n'


        if self._3b:
            for index, line in enumerate(lines):
                if 'k3b_cutoff=' in line:
                    lines[index] = f'k3b_cutoff="{k3b_dict["cutoff"]}"\n'
                if 'k3b_delta=' in line:
                    lines[index] = f'k3b_delta="{k3b_dict["delta"]}"\n'
                if 'k3b_theta=' in line:
                    lines[index] = f'k3b_theta="{k3b_dict["theta_uniform"]}"\n'
        with open('train.sh', 'w') as f:
            f.writelines(lines)




    def initial_sample (self):
        print('='*300)
        print('ITERATION NUMBER (initial sample):', self.num_evaluations + 1)
        print('='*300)

        cutoff3b, delta3b, theta3b, nsparse3b = None, None, None, None

        if self._2b:
            cutoff = np.random.uniform(4, 6)
            delta = np.random.uniform(1, 2.5)
            theta = np.random.uniform(1, 2.5)
            nsparse2b = np.random.randint(10, 20)

            if self._3b:
                cutoff3b = np.random.uniform(3, 5)
                delta3b = np.random.uniform(0.001, 0.1)
                theta3b = np.random.uniform(1, 2.5)
                nsparse3b = np.random.randint(50, 300)


        if self._2b:
            print(f"Random trial params (2b) (cutoff, delta, theta_uniform): {cutoff:.2f}, {delta:.2f}, {theta:.2f}")
        if self._3b:
            print(f"Random trial params (3b) (cutoff, delta, theta_uniform): {cutoff3b:.2f}, {delta3b:.4f}, {theta3b:.2f}")

        # Update the train.sh script with new parameters
        print("Updating train.sh with new params...")
        self.update_train_script(cutoff=cutoff, delta=delta, theta=theta, sparse2b=self.nsparse_2b,
                                cutoff3b=cutoff3b, delta3b=delta3b, theta3b=theta3b, sparse3b=self.nsparse_3b )



    def bayesian_opt (self):

        self.F = torch.tensor([], dtype=torch.double)
        self.train_X = torch.tensor([], dtype=torch.double).reshape(0, 8) if self._3b else torch.tensor([], dtype=torch.double).reshape(0, 4)
        for i in range (len(self.param_data['cutoff2b'])):
            hyperparams = []
            if self._2b:
                print(self.param_data['cutoff2b'][i])
                hyperparams.append(torch.tensor([self.param_data['cutoff2b'][i]], dtype = torch.double))
                hyperparams.append(torch.tensor([self.param_data['delta2b'][i]], dtype = torch.double))
                hyperparams.append(torch.tensor([self.param_data['theta2b'][i]], dtype = torch.double))
                hyperparams.append(torch.tensor([self.param_data['sparse2b'][i]], dtype = torch.double))

                if self._3b:
                    hyperparams.append(torch.tensor([self.param_data['cutoff3b'][i]], dtype = torch.double))
                    hyperparams.append(torch.tensor([self.param_data['delta3b'][i]], dtype = torch.double))
                    hyperparams.append(torch.tensor([self.param_data['theta3b'][i]], dtype = torch.double))
                    hyperparams.append(torch.tensor([self.param_data['sparse3b'][i]], dtype = torch.double))

            X = torch.tensor(hyperparams, dtype=torch.double)


            self.train_X = torch.cat((self.train_X, X.unsqueeze(0)))

            force = torch.tensor([self.param_data['fmae'][i]], dtype = torch.double)
            self.F = torch.cat((self.F, force), dim = -1)

        print('Training set of hyperparameters:', self.train_X)
        print('Objective function evaluations:', self.F)

        print('Training Gaussian Process...')
        gp = SingleTaskGP(
            train_X=self.train_X,
            train_Y=-self.F.unsqueeze(-1),  # negative of the objective function ensures this is a minimisation problem
            input_transform=Normalize(d=8 if self._3b else 4),  # CHANGE DIMENSIONS FOR MORE PARAMS
            outcome_transform=Standardize(m=1),
        )

        # Compute mll of GP given data
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        # GP model is fitted to the data by maxmising mll
        fit_gpytorch_mll(mll)

        # Initialise acquisition function and pass best value of RMSE
        self.best_F = -self.F.max()
        logEI = LogExpectedImprovement(model=gp, best_f=self.best_F)

        if self._2b and not self._3b:
            # Reshape bounds to (2, d), where d=3 (Cutoff, delta, theta bounds)
            bounds = torch.tensor([[4, 1, 1, 10],    # Lower bounds
                           [6, 2.5, 2.5, 80]])  # Upper bounds
        elif self._2b and self._3b:
            # Reshape bounds to (2, d), where d=6 (Cutoff, delta, theta for both 2b and 3b)
            bounds = torch.tensor([[4, 1, 1, 10, 3, 0.001, 1, 50],   # Lower bounds
                                [6, 2.5, 2.5, 80, 5, 0.1, 2.5, 300]])  # Upper bounds



        print('Optimising acquisition function...')

        # Optimise the acquisition function and return 1 candidate tensor
        candidate, _ = optimize_acqf(
            logEI, bounds=bounds, q=1, num_restarts=20, raw_samples=512,
        )

        print('New candidate:',candidate)

        print('Updating train.sh with new params...')

        if self._2b and not self._3b:
            self.update_train_script(cutoff = candidate[0, 0].item(), delta = candidate[0, 1].item(), theta = candidate[0, 2].item(),
                                     sparse2b=round(candidate[0, 3].item()), cutoff3b=None, delta3b=None,
                                  theta3b = None, sparse3b=None)
        if self._2b and self._3b:
            self.update_train_script(cutoff = candidate[0, 0].item(), delta = candidate[0, 1].item(),
                                  theta = candidate[0, 2].item(), sparse2b=round(candidate[0, 3].item()), cutoff3b=candidate[0, 4].item(),
                                  delta3b=candidate[0, 5].item(), theta3b = candidate[0, 6].item(), sparse3b=round(candidate[0, 7].item()))






        # EXIT ONCE DONE GO TO BASH SCRIPT



    def run_BO (self):



        print('Updating params.txt...')
        self.update_param_file()
        print('params.txt updated!')

        # Store existing params in memory
        self.read_params()

        # Don't begin BO unless initial samples criteria has been met (init)
        if self.num_evaluations < self.initial_samples:
            # Draw initial random samples, write to train.sh
            self.initial_sample()
            print('Exiting script, starting training with train.sh')
            exit()
        else:
            # Check if optimisation has converged
            self.check_convergence()
            if self.converged == False:
                print('='*300)
                print('ITERATION NUMBER (Bayesian optimisation):', self.num_evaluations - self.initial_samples + 1)
                print('='*300)
                # Run Bayesian optimisation using existing data in params.txt and update train.sh
                self.bayesian_opt()
                print('Exiting script, starting training with train.sh')
            else:
                # Create marker file to detect with bash script as flag to end simulations
                with open ('convergence_complete.txt', 'w') as f:
                    f.write('a')
                print('Convergence complete, exiting...')
            exit()



    def run (self):
        # Run newly trained potential against validation databse
        self.validate_potential()

        # Situation where sparse sampling is complete so we begin BO
        self.run_BO()




Bayesian_Optimisation().run()
