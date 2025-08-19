#!/bin/bash

# Input File
n='1'
infile="train_filtered.xyz"
outfile="gp${n}.xml"

# Gap-fit Settings
settings="sparse_jitter=1e-8 \
          default_sigma={0.002 0.02 0.02 0.0} \
          e0={Ba:-691.598:O:-430.085:Y:-1038.579:Cu:-1306.079} \
          config_type_sigma={YBCO7:0.002:0.02:0.02:0.0:YBCO6:0.002:0.02:0.02:0.0:Y2O3:0.005:0.05:0.05:0.0:BaO:0.005:0.05:0.05:0.0:Cu2O:0.005:0.05:0.05:0.0:O2:0.005:0.05:0.05:0.0} \
          core_param_file=pairpot.xml \
          core_ip_args={IP Glue} \
          sparse_separate_file=F \
          do_copy_at_file=F \
          energy_parameter_name=dft_energy \
          force_parameter_name=dft_force"

# Two-body descriptors with gaussian kernel
k2b_params="distance_2b cutoff=5.475287779771019 cutoff_transition_width=1.0 delta=1.5988954444172057 n_sparse=20 sparse_method=uniform covariance_type=ARD_SE add_species=T theta_uniform=1.4866572134311494"

k3b_params="angle_3b cutoff=4.85899923760861 theta_uniform=2.151713890856283 covariance_type=ARD_SE n_sparse=50 delta=0.01977635928288668 add_species=T"


# Run the Program
exec="/work/e05/e05/ash141/codes/QUIP2/QUIP/build/archer2_mpich+openmp/gap_fit"
$exec atoms_filename=$infile gap={$k2b_params:$k3b_params} $settings gp_file=$outfile
