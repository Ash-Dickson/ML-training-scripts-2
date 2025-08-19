class File_constructor:

    def __init__ (self, ermse, frmse, emae, fmae, file_name='train.sh', _2b=True, _3b=False):

        with open(file_name, 'r') as f:
            self.content = f.readlines()

        self._2b = _2b
        self._3b = _3b

        self.E_rmse = ermse
        self.F_rmse = frmse
        self.E_mae = emae
        self.F_mae = fmae

    def get_params(self):

        k2b_dict = {}
        k3b_dict = {}
        for line in self.content:

            if 'k2b_params=' in line:

                params = line.split('"')[1].strip('"').strip('distance_2b').split()

                for param in params:
                    key, value = param.split('=')
                    k2b_dict[key] = value
                self.cutoff2b = k2b_dict['cutoff']
                self.theta_uniform2b = k2b_dict['theta_uniform']
                self.delta2b = k2b_dict['delta']
                self.sparse2b = k2b_dict['n_sparse']

            if 'k3b_params=' in line:

                params = line.split('"')[1].strip('"').strip('angle_3b').split()
                for param in params:
                    key, value = param.split('=')
                    k3b_dict[key] = value
                self.cutoff3b = k3b_dict['cutoff']
                self.theta_uniform3b = k3b_dict['theta_uniform']
                self.delta3b = k3b_dict['delta']
                self.sparse3b = k3b_dict['n_sparse']

        return k2b_dict, k3b_dict

    def write_header(self):

        with open('params.txt', 'w') as f:

            header = f"{'Evaluation Number':<20}"
            header_length = 100
            if self._2b:
                header += "{:^20} {:^20} {:^20} {:^20}".format("cutoff_2b", "delta_2b", "theta_uniform_2b", "nsparse2b")
                header_length += 80
            if self._3b:
                header += "{:^20} {:^20} {:^20} {:^20}".format("cutoff_3b", "delta_3b", "theta_uniform_3b", "nsparse3b")
                header_length += 80
            header += "{:^20} {:^20} {:^20} {:^20}".format("Energy_RMSE", "Force_RMSE", "Energy_MAE", "Force_MAE\n")
            f.write(header)
            f.write(("=" * header_length).center(header_length) + "\n")

    def write_sparse_header(self):
        if self._2b:
            with open ('sparse_data2b.txt', 'w') as f:
                    header = "{:^20} {:^20}".format("n_sparse_2b", "MAE ener\n")
                    f.write(header)
                    f.write(("=" * 40).center(40) + "\n")
        if self._3b:
            with open ('sparse_data3b.txt', 'w') as f:
                    header = "{:^20} {:^20}".format("n_sparse_3b", "MAE ener\n")
                    f.write(header)
                    f.write(("=" * 40).center(40) + "\n")

    def write_new_data_sparse(self, nsparse, k2b = False, k3b = False):
        if k2b:
            with open('sparse_data2b.txt', 'a') as f:
                row = "{:^20} {:^20.6f}\n".format(nsparse, self.E_mae)
                f.write(row)
        if k3b:
            with open('sparse_data3b.txt', 'a') as f:
                row = "{:^20} {:^20.6f}\n".format(nsparse, self.E_mae)
                f.write(row)






    def write_new_data(self):

        with open('params.txt', 'r') as f:
            iter_number = len(f.readlines()) - 1


        with open('params.txt', 'a') as f:
            self.get_params()
            row = "{:^20}".format(iter_number)
            if self._2b:
                row += "{:^20} {:^20} {:^20} {:^20}".format(self.cutoff2b,self.delta2b,self.theta_uniform2b, self.sparse2b)
            if self._3b:
                row += "{:^20} {:^20} {:^20} {:^20}".format(self.cutoff3b,self.delta3b,self.theta_uniform3b, self.sparse3b)
            row += "{:^20} {:^20} {:^20} {:^20}\n".format(self.E_rmse, self.F_rmse, self.E_mae, self.F_mae)
            f.write(row)
