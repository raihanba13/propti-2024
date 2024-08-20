import sys
import logging
import numpy as np
import spotpy

from mpi4py import MPI


class FitnessMethodInterface:

    def __init__(self, scale_fitness=True):
        self.scale_fitness = scale_fitness

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        print("using undefined function")


class FitnessMethodRMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined RMSE value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodRMSE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                rmse = self.penalty
                return rmse

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        # Calculate the root mean squared error (RMSE).
        rmse = np.sqrt(((y_e_mapped - y_m_mapped) ** 2).mean())

        # Scale the RMSE value.
        if self.scale_fitness == 'mean' or self.scale_fitness is True:
            # Return the RMSE scaled by the mean value of the data series.
            return rmse / np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse / np.abs(np.max(y_e_mapped) - np.min(y_e_mapped))
        elif self.scale_fitness == 'interquartile':
            # Return the RMSE as is.
            return rmse
        else:
            # Return the RMSE as is.
            return rmse


class FitnessMethodRangeRMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, y_relative_range=None,
                 scale_fitness=True):
        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        if y_relative_range is None:
            self.y_relative_range = 0.05
        else:
            self.y_relative_range = abs(y_relative_range)
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """

        compute x array on which the data sets shall be mapped to in order
        to compute the RMSE on the same definition range

        :param x_e:
        :param y_e:
        :param y2_e:
        :param x_m:
        :param y_m:
        :return:
        """

        msg = "* Compute FitnessMethodRangeRMSE."
        logging.debug(msg)

        if self.x_def is None:
            if self.x_def_range is None:
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n, endpoint=True)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)
        y_rmse = np.zeros(y_e_mapped.shape)
        for i, value in enumerate(y_e_mapped):
            if (y_e_mapped[i]*(1-self.y_relative_range)) <= y_m_mapped[i] <=\
                    (y_e_mapped[i]*(1+self.y_relative_range)):
                y_rmse[i] = 0
            else:
                y_rmse[i] = (y_e_mapped[i] - y_m_mapped[i]) ** 2
        rmse = np.sqrt(y_rmse.mean())
        if self.scale_fitness == 'mean' or self.scale_fitness is True:
            return rmse / np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse / np.abs(np.max(y_e_mapped) - np.min(y_e_mapped))
        elif self.scale_fitness == 'interquartile':
            return rmse
        else:
            return rmse


class FitnessMethodBandRMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True):
        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """

        compute x array on which the data sets shall be mapped to in order to
        compute the RMSE on the same definition range

        :param x_e:
        :param y_e:
        :param y2_e:
        :param x_m:
        :param y_m:
        :return:
        """

        msg = "* Compute FitnessMethodBandRMSE."
        logging.debug(msg)

        if self.x_def is None:
            if self.x_def_range is None:
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n, endpoint=True)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_e_mapped_b2 = np.interp(self.x_def, x_e, y2_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)
        y_rmse = np.zeros(y_e_mapped.shape)
        for i, value in enumerate(y_e_mapped):
            if np.min((y_e_mapped[i], y_e_mapped_b2[i])) <= y_m_mapped[i] <= \
                    np.max((y_e_mapped[i], y_e_mapped_b2[i])):
                y_rmse[i] = 0
            else:
                y_rmse[i] = np.min((((y_e_mapped[i] - y_m_mapped[i]) ** 2),
                                    ((y_e_mapped_b2[i] - y_m_mapped[i]) ** 2)))
        rmse = np.sqrt(y_rmse.mean())
        if self.scale_fitness == 'mean' or self.scale_fitness is True:
            return rmse / np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse / np.abs(np.max(y_e_mapped) - np.min(y_e_mapped))
        elif self.scale_fitness == 'interquartile':
            return rmse
        else:
            return rmse


class FitnessMethodThreshold(FitnessMethodInterface):

    def __init__(self, threshold_type, threshold_target_value=None,
                 threshold_value=None, threshold_range=None,
                 scale_fitness=True):

        super().__init__(scale_fitness)

        threshold_types = ['upper', 'lower', 'range_minmax']
        if threshold_type not in threshold_types:
            print("wrong threshold type, available types are:", threshold_types)
            # TODO handle this?
        self.type = threshold_type
        self.threshold_target_value = threshold_target_value
        self.value = threshold_value
        self.range = threshold_range
        self.scale_fitness = scale_fitness

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """

        :param x_e:
        :param y_e:
        :param y2_e:
        :param x_m:
        :param y_m:
        :return:
        """

        msg = "* Compute FitnessMethodThreshold."
        logging.debug(msg)

        x_e_threshold = None
        x_m_threshold = None
        if self.type == "upper" or self.type == "lower":
            # only needed for experimental data if no target value was specified
            if self.threshold_target_value is None:
                x_e_threshold = self.simple_threshold(self.type,
                                                      self.value,
                                                      x_e,
                                                      y_e)
            else:
                x_e_threshold = self.threshold_target_value
            x_m_threshold = self.simple_threshold(self.type,
                                                  self.value,
                                                  x_m,
                                                  y_m)

        if self.type == "range_minmax":
            # only needed for experimental data if no target value was specified
            if self.threshold_target_value is None:
                x_e_threshold_lower = self.simple_threshold("lower",
                                                            self.range[0],
                                                            x_e, y_e)
                x_e_threshold_upper = self.simple_threshold("upper",
                                                            self.range[1],
                                                            x_e, y_e)
            x_m_threshold_lower = self.simple_threshold("lower",
                                                        self.range[0],
                                                        x_m, y_m)
            x_m_threshold_upper = self.simple_threshold("upper",
                                                        self.range[1],
                                                        x_m, y_m)

            # check if target value was explicitly passed
            if self.threshold_target_value is not None:
                x_e_threshold = self.threshold_target_value
            else:
                # result is the smallest value in x when the range was left
                if x_e_threshold_lower is not None and \
                        x_e_threshold_upper is not None:
                    x_e_threshold = np.min(x_e_threshold_lower,
                                           x_e_threshold_upper)
            if x_m_threshold_lower is not None and \
                    x_m_threshold_upper is not None:
                x_m_threshold = np.min(x_m_threshold_lower, x_m_threshold_upper)

        # check if the experimental data returns a valid threshold evaluation
        if x_e_threshold is None:
            print("ERROR: rethink your fitness method choice")
            logging.error("rethink your fitness method choice")
            sys.exit(1)

        # if the model data never reaches the threshold,
        # return maximal deviation w.r.t. the experimental value,
        # i.e. maximal model x-value minus experimental threshold position
        if x_m_threshold is None:
            x_m_max_distance = np.abs(np.max(x_m) - x_e_threshold)
            if self.scale_fitness:
                return np.abs(x_m_max_distance / x_e_threshold)
            else:
                return x_m_max_distance

        if self.scale_fitness:
            return np.abs((x_e_threshold - x_m_threshold) / x_e_threshold)

        return np.abs(x_e_threshold - x_m_threshold)

    def simple_threshold(self, t, v, x, y):
        """

        :param t:
        :param v:
        :param x:
        :param y:
        :return:
        """
        indices = None
        if t == "upper":
            indices = np.where(y > v)
        if t == "lower":
            indices = np.where(y < v)

        if len(indices[0]) > 0:
            result_index = indices[0][0]
            result_x = x[result_index]
        else:
            print("threshold was not reached")
            result_x = None

        return result_x


class FitnessMethodIntegrate(FitnessMethodInterface):
    """
    Integrate a data series and determine the distance to a target value.
    For instance to get the heat of combustion from MCC data.
    """

    def __init__(self, n_points, x_def_range=None, scale_fitness=True,
                 integrate_factor=1.0):
        """
        Constructor.
        :param n_points: number of data points on which to interpolate the
            data series.
        :param x_def_range:
        :param scale_fitness:
        :param integrate_factor: multiply the integration result, default: 1.0
        """

        msg = "* From FitnessMethodIntegrate.__init__"
        logging.debug(msg)

        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.integrate_factor = integrate_factor
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

        # TODO: implement parameter check in propti_prepare
        if self.n is None:
            msg = "* Note: 'n_points' is None, please choose a number!"
            logging.error(msg)
            # Is supposed to stop the whole MPI job, i.e. communicate
            # "upwards" to the main process that it shuts down.
            comm = MPI.COMM_WORLD
            comm.abort()
            # sys.exit()

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Compute x array on which the data sets shall be mapped to,
        in order to compute the RMSE on the same definition range.
        """

        msg = "* From FitnessMethodIntegrate.compute"
        logging.debug(msg)

        if self.x_def is None:
            msg = "* Note: 'x_def' is None."
            logging.debug(msg)
            if self.x_def_range is None:
                msg = "* Note: 'x_def_range' is None."
                logging.debug(msg)
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
                msg = "* From 'compute': 'x_def_range' is now: {}."
                logging.debug(msg.format(self.x_def_range))

            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n,
                                     endpoint=True)
            msg = "* From 'compute': 'x_def' is now: {}."
            logging.debug(msg.format(self.x_def))

        msg = "* Mapping data..."
        logging.debug(msg)

        # Map data series to the same definition range.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)
        msg = "* FitnessMethodIntegrate.compute: y_e_mapped={}, y_m_mapped={}"
        logging.debug(msg.format(y_e_mapped, y_m_mapped))

        # Integrate experiment and model data series.
        value_e = np.trapz(y_e_mapped, self.x_def) * self.integrate_factor
        value_m = np.trapz(y_m_mapped, self.x_def) * self.integrate_factor
        msg = "* FitnessMethodIntegrate.compute: value_e={}, value_m={}"
        logging.debug(msg.format(value_e, value_m))

        # Compare experiment and model data.
        rmse = np.abs(value_e - value_m)

        msg = "* FitnessMethodIntegrate.compute: value_e={}, value_m={}, rmse={}"
        logging.debug(msg.format(value_e, value_m, rmse))

        # Scale the fitness value, if required.
        # TODO: Find better way for scaling
        if self.scale_fitness is True:
            # return rmse / np.abs(np.mean(y_e_mapped))
            return rmse / value_e
        # elif self.scale_fitness == 'minmax':
        #     return rmse / np.abs(y_e_mapped[-1] - y_e_mapped[0])
        # elif self.scale_fitness == 'interquartile':
        #     return rmse
        else:
            return rmse


"""
https://github.com/thouska/spotpy/blob/master/src/spotpy/objectivefunctions.py
* BIAS
* PBIAS
* NASHSUTCLIFFE
* LOGNASHSUTCLIFFE
* LOGP
* CORELATIONCOEFFICIENT
* RSQUARED
* MSE
* MAE
* RELATIVERMSE
* AGREEMENTINDEX
* COVARIANCE
* DECOMPOSEDMSE
* KGE
* KGENONPARAMETRIC
* RSR
* VOLUMEERROR
"""

class FitnessMethodBIAS(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodBIAS."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                bias = self.penalty
                return bias

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        bias = np.nansum(y_e_mapped - y_m_mapped) / len(y_m_mapped)

        return float(bias)

        # TODO understand scaling and do we need it for this method?


class FitnessMethodPBIAS(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodPBIAS."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                pbias = self.penalty
                return pbias

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        # Calculate the root mean squared error (RMSE).
        pbias = 100 * (float(np.nansum(y_m_mapped - y_e_mapped)) / float(np.nansum(y_e_mapped)))
        
        return float(pbias)
    

class FitnessMethodNASHSUTCLIFFE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodNASHSUTCLIFFE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                nse = self.penalty
                return nse

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)
        mean_observed = np.nanmean(y_e_mapped)

        numerator = np.nansum((y_e_mapped - y_m_mapped) ** 2)
        denominator = np.nansum((y_e_mapped - mean_observed) ** 2)
        # compute coefficient
        return 1 - (numerator / denominator)
    

class FitnessMethodLOGNASHSUTCLIFFE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05, epsilon2=0.0):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference
        self.epsilon2 = epsilon2

        """
        :epsilon2: Value which is added to simulation and evaluation data to errors when simulation or evaluation data has zero values
        :type: float or list
        """

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodLOGNSE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                lognse = self.penalty
                return lognse

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        # s => simulation, e => evaluation/ experimental/ observed
        s, e = np.array(y_m_mapped) + self.epsilon2, np.array(y_e_mapped) + self.epsilon2

        return float(
            1
            - sum((np.log(s) - np.log(e)) ** 2)
            / sum((np.log(e) - np.mean(np.log(e))) ** 2)
        )



class FitnessMethodLOGP(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Logarithmic probability distribution, log_p

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodLOGP."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                logp = self.penalty
                return logp

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        """
        scale = np.mean(y_e_mapped) / 10
        if scale < 0.01:
            scale = 0.01

        y = (y_e_mapped - y_m_mapped) / scale
        normpdf = -(y**2) / 2 - np.log(np.sqrt(2 * np.pi))
        return np.mean(normpdf)
        """

        # Calculate the likelihood
        log_p = spotpy.objectivefunctions.log_p(y_e_mapped, y_m_mapped)

        return log_p



class FitnessMethodCORELATIONCOEFFICIENT(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodCORELATIONCOEFFICIENT."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                correlation_coefficient = self.penalty
                return correlation_coefficient

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        correlation_coefficient = np.corrcoef(y_e_mapped, y_m_mapped)[0, 1]

        return correlation_coefficient



class FitnessMethodRSQUARED(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodRSQUARED."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                r_squared = self.penalty
                return r_squared

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        correlation_coefficient = np.corrcoef(y_e_mapped, y_m_mapped)[0, 1]
        
        return correlation_coefficient ** 2

              

class FitnessMethodMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodMSE between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodMSE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                mse = self.penalty
                return mse

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        mse = np.nanmean((y_e_mapped - y_m_mapped) ** 2)  

        return mse



class FitnessMethodMAE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodMAE between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodMAE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                mae = self.penalty
                return mae

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        mae = np.mean(np.abs(y_m_mapped - y_e_mapped)) 

        return mae
    


class FitnessMethodRELATIVERMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        # Instantiate FitnessMethodRMSE
        self.rmse = FitnessMethodRMSE(
            n_points=n_points,
            x_def_range=x_def_range,
            scale_fitness=scale_fitness,
            check_model_length=check_model_length,
            penalty=penalty,
            difference=difference
        )

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the root mean squared error between two data series.

        This method calculates the root mean squared error (RMSE) between
        two data series. It can also scale the RMSE value based on different
        aspects of the experiment data.
        Furthermore, it can check if the end of the experiment and model
        x-values are close together in an effort to find data series as a
        result of premature model termination, e.g. numerical instabilites
        with FDS. This step is necessary in some cases, where the simulation
        crashes but still some data points are written. Primarily, because
        this method scales the x-range to map both data series to it,
        which allows proper comparison in the RMSE step, when no explicit
        x-range or x-def were provided. The RMSE values are then manually
        set to a high enough value (penalty) to nudge the optimiser away
        from this parameter set.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodRELATIVERMSE."
        logging.debug(msg)

        rmse = self.rmse.compute(x_e, y_e, y2_e, x_m, y_m)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        rrmse = rmse / np.mean(y_e_mapped)

        return rrmse



class FitnessMethodAGREEMENTINDEX(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodAGREEMENTINDEX between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodAGREEMENTINDEX."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                agreement_index = self.penalty
                return agreement_index

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        agreement_index = spotpy.objectivefunctions.agreementindex(y_e_mapped, y_m_mapped) 

        return agreement_index      



class FitnessMethodCOVARIANCE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodCOVARIANCE between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodCOVARIANCE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                covariance = self.penalty
                return covariance

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        covariance = spotpy.objectivefunctions.covariance(y_e_mapped, y_m_mapped) 

        return covariance  



class FitnessMethodDECOMPOSEDMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodDECOMPOSEDMSE between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodDECOMPOSEDMSE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                decomposed_mse = self.penalty
                return decomposed_mse

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        decomposed_mse = spotpy.objectivefunctions.decomposed_mse(y_e_mapped, y_m_mapped) 

        return decomposed_mse        
    


class FitnessMethodKGE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodKGE between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodKGE."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                kge = self.penalty
                return kge

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        kge = spotpy.objectivefunctions.kge(y_e_mapped, y_m_mapped, return_all=False) 

        return kge 



class FitnessMethodKGENONPARAMETRIC(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodKGENONPARAMETRIC between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodKGENONPARAMETRIC."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                kge_non_parametric = self.penalty
                return kge_non_parametric

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        kge_non_parametric = spotpy.objectivefunctions.kge_non_parametric(y_e_mapped, y_m_mapped, return_all=False) 

        return kge_non_parametric  

               
    


class FitnessMethodRSR(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """

        RMSE-observations standard deviation ratio
            
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        # Instantiate FitnessMethodRMSE
        self.rmse = FitnessMethodRMSE(
            n_points=n_points,
            x_def_range=x_def_range,
            scale_fitness=scale_fitness,
            check_model_length=check_model_length,
            penalty=penalty,
            difference=difference
        )

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodRSR between two data series.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodRSR."
        logging.debug(msg)

        rmse = self.rmse.compute(x_e, y_e, y2_e, x_m, y_m)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        rsr = rmse / np.std(y_e_mapped)

        return rsr
    


class FitnessMethodVOLUMEERROR(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the FitnessMethodVOLUMEERROR between two data series.
        Using spotpy.objectivefunctions.volume_error.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute FitnessMethodVOLUMEERROR."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                volume_error = self.penalty
                return volume_error

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        volume_error = spotpy.objectivefunctions.volume_error(y_e_mapped, y_m_mapped) 

        return volume_error 




class FitnessMethodETEVSF(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the ExponentialTransformErrVarShapingFactor between two data series.
        Using spotpy.likelihood.ExponentialTransformErrVarShapingFactor.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute ExponentialTransformErrVarShapingFactor."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                volume_error = self.penalty
                return volume_error

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        etevsf = spotpy.likelihoods.ExponentialTransformErrVarShapingFactor(y_e_mapped, y_m_mapped) 

        return etevsf 
    


class FitnessMethodIEVSF(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the InverseErrorVarianceShapingFactor between two data series.
        Using spotpy.likelihoods.InverseErrorVarianceShapingFactor.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute InverseErrorVarianceShapingFactor."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                volume_error = self.penalty
                return volume_error

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        ievsf = spotpy.likelihoods.InverseErrorVarianceShapingFactor(y_e_mapped, y_m_mapped) 

        return ievsf 



class FitnessMethodGLMEO(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the gaussianLikelihoodMeasErrorOut between two data series.
        Using spotpy.likelihoods.gaussianLikelihoodMeasErrorOut.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute gaussianLikelihoodMeasErrorOut."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                volume_error = self.penalty
                return volume_error

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        glmeo = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(y_e_mapped, y_m_mapped) 

        return glmeo 



class FitnessMethodlogLikelihood(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True,
                 check_model_length=True, penalty=3.5, difference=0.05):
        #TODO set default of check_model_length to False
        """
        Constructor, setting up basic parameters.

        :param n_points: number of evenly spaced data points, endpoint included
        :param x_def_range:
        :param scale_fitness: default=True
        :param check_model_length: flag to control if a check for premature
            model termination is to be conducted.
        :param penalty: pre-defined BIAS value that is high enough to nudge
            the optimiser away from this parameter set.
        :param difference: percentage that is used to calculate the lower limit.
        """

        self.n_points = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        self.check_model_length = check_model_length
        self.penalty = penalty
        self.difference = difference

        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        """
        Calculates the logLikelihood between two data series.
        Using spotpy.likelihoods.logLikelihood.

        :param x_e: x-values of the experiment data
        :param y_e: y-values of the experiment data
        :param y2_e: y-values of the experiment data
        :param x_m: x-values of the model data
        :param y_m: y-values of the model data
        :return: root mean squared error, possibly scaled
        """

        msg = "* Compute logLikelihood."
        logging.debug(msg)

        # Check for premature model termination.
        if self.check_model_length is True:
            # Determine max x-value.
            x_e_max = np.max(x_e)  # x_e[-1]
            x_m_max = np.max(x_m)
            # Calculate lower limit.
            epsilon = self.difference * x_e_max
            threshold = x_e_max - epsilon

            # Check if the model x-values are below the lower limit.
            if x_m_max < threshold:
                # Award a penalty.
                volume_error = self.penalty
                return volume_error

        # Determine the length of the
        if self.x_def is None:
            if self.x_def_range is None:
                # Find minimum and maximum values that are in both data series.
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]

            # Create an equidistant interval over n_points, including the
            # endpoint.
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n_points,
                                     endpoint=True)

        # Map both y data series on the same x-values, to allow for a
        # meaningful comparison.
        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        logLikelihood = spotpy.likelihoods.logLikelihood(y_e_mapped, y_m_mapped) 

        return logLikelihood 

