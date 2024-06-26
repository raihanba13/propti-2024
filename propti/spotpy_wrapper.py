import logging
import sys
import os
import shutil
import numpy as np
import tempfile
import os.path
from pathlib import Path
import spotpy

from .data_structures import Parameter, ParameterSet, SimulationSetup, \
    SimulationSetupSet, Relation, OptimiserProperties

from .basic_functions import create_input_file, run_simulations, \
    extract_simulation_data


####################
# SPOTPY SETUP CLASS

class SpotpySetup(object):
    def __init__(self,
                 params: ParameterSet,
                 setups: SimulationSetupSet,
                 optimiser: OptimiserProperties):

        self.setups = setups
        self.params = params
        self.optimiser = optimiser
        self.spotpy_parameter = []
        self.j=0

        for p in params:
            logging.debug("Setup SPOTPY parameter: {}".format(p.name))
            if p.distribution == 'uniform':

                optguess = None
                step = None
                if p.value is not None:
                    optguess = p.value
                if p.max_increment is not None:
                    step = p.max_increment

                cp = spotpy.parameter.Uniform(p.place_holder,
                                              p.min_value, p.max_value,
                                              step=step,
                                              optguess=optguess,
                                              minbound=p.min_value,
                                              maxbound=p.max_value)
                self.spotpy_parameter.append(cp)
            else:

                logging.error(
                    '* Parameter distribution function unknown: {}'.format(
                        p.distribution))

    def parameters(self):
        return spotpy.parameter.generate(self.spotpy_parameter)

    def simulation(self, vector):
        logging.debug("* Current SPOTPY simulation vector: {}".format(vector))

        # Copy SPOTPY parameter vector to parameter set.
        for i in range(len(vector)):
            self.params[i].value = vector[i]

        # Update all simulation setup parameter sets.
        for s in self.setups:
            s.model_parameter.update(self.params)

        # Create run directories for all simulation setups.
        for s in self.setups:
            if s.execution_dir_prefix:
                tmp_dir_root = s.execution_dir_prefix
            else:
                tmp_dir_root = os.path.join(os.getcwd(), s.work_dir)
            s.execution_dir = tempfile.mkdtemp(prefix='rundir_',
                                               dir=tmp_dir_root)
            create_input_file(s)

        # Run all simulations.
        logging.debug("* Run all simulations.")
        run_simulations(self.setups, self.optimiser.num_subprocesses)

        # gather simulation data
        for s in self.setups:
            logging.debug("* Start data extraction.")
            extract_simulation_data(s)
        logging.debug("* Finished data extraction.")

        # Clean up temporary execution directories.
        for s in self.setups:
            logging.debug("* Clean up of temporary execution directories.")
            shutil.rmtree(s.execution_dir)

        # Initialise values needed to compute fitness.
        logging.debug("* Compute fitness values.")
        global_fitness_value = 0
        individual_fitness_values = list()

        # Compute fitness values.
        for s in self.setups:
            for r in s.relations:
                logging.debug("* Relation ID: {}.".format(r.id_label))
                current_fitness = r.fitness_weight * r.compute_fitness()
                global_fitness_value += current_fitness
                individual_fitness_values.append(current_fitness)
        self.j+=1
        # first element of returned list is the global fitness value
        # note: in general this should be the simulation data, which is returned
        # due to our data structure, the passing of the fitness values, i.e. result
        # of the objective function, is most convenient approach here
        # last element of the list counts the number of executed simulations
        return [global_fitness_value] + individual_fitness_values + [self.j]

    def evaluation(self):
        logging.debug("* evaluation")
        for s in self.setups:
            for r in s.relations:
                r.read_data(wd='.', target='experiment')

        # Return dummy data.
        # TODO: reconsider returning proper values
        return [1]

    def objectivefunction(self, simulation, evaluation, params):

        # The simulation function does not return simulation data,
        # but directly the fitness values, just pass these values.
        fitness_value = simulation

        msg = "* From objectivefunction: fitness_value={}"
        logging.debug(msg.format(fitness_value))

        # return the global fitness value instead of list
        return fitness_value[0]


def run_optimisation(params: ParameterSet,
                     setups: SimulationSetupSet,
                     opt: OptimiserProperties) -> ParameterSet:
    spot = SpotpySetup(params, setups, opt)
    # Check if a break file exists for restarting.
    break_file_name = Path('{}.break'.format(opt.db_name))
    break_point = 'write'
    if break_file_name.is_file():
        break_point = 'readandwrite'
    parallel = 'seq'
    if opt.mpi:
        parallel = 'mpi'
    if opt.algorithm == 'sceua':
        sampler = spotpy.algorithms.sceua(spot,
                                          dbname=opt.db_name,
                                          dbformat=opt.db_type,
                                          parallel=parallel,
                                          db_precision=np.float64,
                                          breakpoint=break_point,
                                          backup_every_rep=opt.backup_every)

        ngs = opt.ngs
        if not ngs:
            ngs = len(params)
            # Set amount of parameters as default for number of complexes
            # if not explicitly specified.
            opt.ngs = ngs
        results = sampler.sample(opt.repetitions, ngs=ngs,
                                 max_loop_inc=opt.max_loop_inc)
        # results = sampler.sample(opt.repetitions, ngs=ngs)
    elif opt.algorithm == 'fscabc':
        # print(break_point)
        # breakpoint()
        sampler = spotpy.algorithms.fscabc(spot,
                                           dbname=opt.db_name,
                                           dbformat=opt.db_type,
                                           parallel=parallel,
                                           db_precision=np.float64,
                                        #    breakpoint=break_point,
                                           breakpoint=None,
                                           backup_every_rep=opt.backup_every)
        eb = opt.eb
        if not eb:
            eb = 48
            # Set amount of parameters as default for number of complexes
            # if not explicitly specified.
            opt.eb = eb
        results = sampler.sample(opt.repetitions, eb=eb)
    elif opt.algorithm == 'abc':
        sampler = spotpy.algorithms.abc(spot,
                                        dbname=opt.db_name,
                                        dbformat=opt.db_type,
                                        parallel=parallel,
                                        breakpoint=break_point,
                                        backup_every_rep=opt.backup_every)
        eb = opt.eb
        if not eb:
            eb = 48
            # Set amount of parameters as default for number of complexes
            # if not explicitly specified.
            opt.eb = eb
        # results = sampler.sample(opt.repetitions, eb=eb)
        results = sampler.sample(opt.repetitions)
    elif opt.algorithm == 'mc':
        sampler = spotpy.algorithms.mc(spot,
                                       dbname=opt.db_name,
                                       dbformat=opt.db_type,
                                       parallel=parallel)
        results = sampler.sample(opt.repetitions)

    elif opt.algorithm == 'dream':
        sampler = spotpy.algorithms.dream(spot,
                                          dbname=opt.db_name,
                                          dbformat=opt.db_type,
                                          parallel=parallel)
        results = sampler.sample(opt.repetitions)

    elif opt.algorithm == 'demcz':
        sampler = spotpy.algorithms.demcz(spot,
                                          dbname=opt.db_name,
                                          dbformat=opt.db_type,
                                          parallel=parallel)
        results = sampler.sample(opt.repetitions)
    elif opt.algorithm == 'mcmc':
        sampler = spotpy.algorithms.mcmc(spot,
                                         dbname=opt.db_name,
                                         dbformat=opt.db_type,
                                         parallel=parallel)
        results = sampler.sample(opt.repetitions)
    elif opt.algorithm == 'mle':
        sampler = spotpy.algorithms.mle(spot,
                                        dbname=opt.db_name,
                                        dbformat=opt.db_type,
                                        parallel=parallel)
                                        # breakpoint=break_point,
                                        # backup_every_rep=opt.backup_every)
        results = sampler.sample(opt.repetitions)

    elif opt.algorithm == 'sa':
        sampler = spotpy.algorithms.sa(spot,
                                       dbname=opt.db_name,
                                       dbformat=opt.db_type,
                                       parallel=parallel)
        results = sampler.sample(opt.repetitions)
    elif opt.algorithm == 'rope':
        sampler = spotpy.algorithms.rope(spot,
                                         dbname=opt.db_name,
                                         dbformat=opt.db_type,
                                         parallel=parallel)
        results = sampler.sample(opt.repetitions)

    elif opt.algorithm == 'mc':
        sampler = spotpy.algorithms.mc(spot,
                                       dbname=opt.db_name,
                                       dbformat=opt.db_type,
                                       parallel=parallel)
        results = sampler.sample(opt.repetitions)

    elif opt.algorithm == 'fast':
        sampler = spotpy.algorithms.fast(spot,
                                         dbname=opt.db_name,
                                         dbformat='csv',
                                         parallel=parallel,
                                         breakpoint=break_point,
                                         backup_every_rep=opt.backup_every)
        results = sampler.sample(opt.repetitions)

    elif opt.algorithm == 'lhs':
        sampler = spotpy.algorithms.lhs(spot,
                                         dbname=opt.db_name,
                                         dbformat='csv',
                                         parallel=parallel,
                                         breakpoint=break_point,
                                         backup_every_rep=opt.backup_every)
        results = sampler.sample(opt.repetitions)

    else:
        return(print('No valid optimization algorithm selected'))

    if sampler.status.optimization_direction == 'minimize':
        pars = sampler.status.params_min
    elif sampler.status.optimization_direction == 'maximize':
        pars = sampler.status.params_max
    for i in range(len(params)):
        params[i].value = pars[i]
    for s in setups:
        s.model_parameter.update(params)
    return params


def test_spotpy_setup():
    p1 = Parameter("density", "RHO", min_value=1.0, max_value=2.4,
                   distribution='uniform')
    p2 = Parameter("cp", place_holder="CP", min_value=4.0, max_value=7.2,
                   distribution='uniform')

    ps = ParameterSet()
    ps.append(p1)
    ps.append(p2)

    spot = SpotpySetup(ps)

    for p in spot.parameter:
        print(p.name, p.rndargs)


def test_spotpy_run():
    p1 = Parameter("ambient temperature", place_holder="TMPA", min_value=0,
                   max_value=100,
                   distribution='uniform', value=0)

    ps = ParameterSet()
    ps.append(p1)

    r1 = Relation()
    r1.model[0].label_x = "Time"
    r1.model[0].label_y = "TEMP"
    r1.model[0].file_name = 'TEST_devc.csv'
    r1.model[0].header_line = 1

    r1.experiment[0].x = np.linspace(0, 10, 20)
    r1.experiment[0].y = np.ones_like(r1.experiment[0].x) * 42.1

    r1.x_def = np.linspace(3.0, 8.5, 3)
    relations = [r1]

    s0 = SimulationSetup(name='ambient run',
                         work_dir='test_spotpy',
                         model_template=os.path.join('templates',
                                                     'template_basic_03.fds'),
                         model_executable='fds',
                         relations=relations,
                         model_parameter=ps
                         )
    setups = SimulationSetupSet()
    setups.append(s0)

    for s in setups:
        if not os.path.exists(s.work_dir):
            os.mkdir(s.work_dir)

    run_optimisation(ps, setups)


if __name__ == "__main__":
    # test_spotpy_setup()
    test_spotpy_run()
