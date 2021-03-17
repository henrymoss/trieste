# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains acquisition rules, which choose the optimal point(s) to query on each step of
the Bayesian optimization process.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import tensorflow as tf
from typing_extensions import Final

from ..data import Dataset
from ..models import ProbabilisticModel
from ..space import Box, SearchSpace
from ..type import TensorType
from . import _optimizer
from .function import (
	AcquisitionFunction,
	AcquisitionFunctionBuilder,
	BatchAcquisitionFunction,
	BatchAcquisitionFunctionBuilder,
	ExpectedImprovement,
)

S = TypeVar("S")
""" Unbound type variable. """

SP = TypeVar("SP", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`SearchSpace`. """


class AcquisitionRule(ABC, Generic[S, SP]):
	""" The central component of the acquisition API. """

	@abstractmethod
	def acquire(
		self,
		search_space: SP,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: S | None,
	) -> tuple[TensorType, S]:
		"""
		Return the optimal points within the specified ``search_space``, where optimality is defined
		by the acquisition rule.

		**Type hints:**
		  - The global search space must be a :class:`~trieste.space.SearchSpace`. The exact type
			of :class:`~trieste.space.SearchSpace` depends on the specific
			:class:`AcquisitionRule`.
		  - Each :class:`AcquisitionRule` must define the type of its corresponding acquisition
			state (if the rule is stateless, this type can be `None`). The ``state`` passed
			to this method, and the state returned, must both be of that type.

		:param search_space: The global search space over which the optimization problem
			is defined.
		:param datasets: The known observer query points and observations for each tag.
		:param models: The model to use for each :class:`~trieste.data.Dataset` in ``datasets``
			(matched by tag).
		:param state: The acquisition state from the previous step, if there was a previous step,
			else `None`.
		:return: The optimal points and the acquisition state for this step.
		"""


OBJECTIVE: Final[str] = "OBJECTIVE"
"""
A tag typically used by acquisition rules to denote the data sets and models corresponding to the
optimization objective.
"""


class EfficientGlobalOptimization(AcquisitionRule[None, SearchSpace]):
	""" Implements the Efficient Global Optimization, or EGO, algorithm. """

	def __init__(self, builder: AcquisitionFunctionBuilder | None = None):
		"""
		:param builder: The acquisition function builder to use.
			:class:`EfficientGlobalOptimization` will attempt to **maximise** the corresponding
			acquisition function. Defaults to :class:`~trieste.acquisition.ExpectedImprovement`
			with tag `OBJECTIVE`.
		"""
		if builder is None:
			builder = ExpectedImprovement().using(OBJECTIVE)

		self._builder = builder

	def __repr__(self) -> str:
		""""""
		return f"EfficientGlobalOptimization({self._builder!r})"

	def acquire(
		self,
		search_space: SearchSpace,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: None = None,
	) -> tuple[TensorType, None]:
		"""
		Return the query point that optimizes the acquisition function produced by `builder` (see
		:meth:`__init__`).

		:param search_space: The global search space over which the optimization problem
			is defined.
		:param datasets: The known observer query points and observations.
		:param models: The models of the specified ``datasets``.
		:param state: Unused.
		:return: The single point to query, and `None`.
		"""
		acquisition_function = self._builder.prepare_acquisition_function(datasets, models)
		point = _optimizer.optimize(search_space, acquisition_function)
		return point, None


class ThompsonSampling(AcquisitionRule[None, SearchSpace]):
	""" Implements Thompson sampling for choosing optimal points. """

	def __init__(self, num_search_space_samples: int, num_query_points: int):
		"""
		:param num_search_space_samples: The number of points at which to sample the posterior.
		:param num_query_points: The number of points to acquire.
		"""
		if not num_search_space_samples > 0:
			raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

		if not num_query_points > 0:
			raise ValueError(
				f"Number of query points must be greater than 0, got {num_query_points}"
			)

		self._num_search_space_samples = num_search_space_samples
		self._num_query_points = num_query_points

	def __repr__(self) -> str:
		""""""
		return f"ThompsonSampling({self._num_search_space_samples!r}, {self._num_query_points!r})"

	def acquire(
		self,
		search_space: SearchSpace,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: None = None,
	) -> tuple[TensorType, None]:
		"""
		Sample `num_search_space_samples` (see :meth:`__init__`) points from the
		``search_space``. Of those points, return the `num_query_points` points at which
		random samples yield the **minima** of the model posterior.

		:param search_space: The global search space over which the optimization problem
			is defined.
		:param datasets: Unused.
		:param models: The model of the known data. Uses the single key `OBJECTIVE`.
		:param state: Unused.
		:return: The `num_query_points` points to query, and `None`.
		:raise ValueError: If ``models`` do not contain the key `OBJECTIVE`, or it contains any
			other key.
		"""
		if models.keys() != {OBJECTIVE}:
			raise ValueError(
				f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
			)

		nqp, ns = self._num_query_points, self._num_search_space_samples
		query_points = search_space.sample(ns)  # [ns, ...]
		samples = models[OBJECTIVE].sample(query_points, nqp)  # [nqp, ns, ...]
		samples_2d = tf.reshape(samples, [nqp, ns])  # [nqp, ns]
		indices = tf.math.argmin(samples_2d, axis=1)
		unique_indices = tf.unique(indices).y
		return tf.gather(query_points, unique_indices), None


class TrustRegion(AcquisitionRule["TrustRegion.State", Box]):
	""" Implements the *trust region* acquisition algorithm. """

	@dataclass(frozen=True)
	class State:
		""" The acquisition state for the :class:`TrustRegion` acquisition rule. """

		acquisition_space: Box
		""" The search space. """

		eps: TensorType
		"""
		The (maximum) vector from the current best point to each bound of the acquisition space.
		"""

		y_min: TensorType
		""" The minimum observed value. """

		is_global: bool | TensorType
		"""
		`True` if the search space was global, else `False` if it was local. May be a scalar boolean
		`TensorType` instead of a `bool`.
		"""

		def __deepcopy__(self, memo: dict[int, object]) -> TrustRegion.State:
			box_copy = copy.deepcopy(self.acquisition_space, memo)
			return TrustRegion.State(box_copy, self.eps, self.y_min, self.is_global)

	def __init__(
		self,
		builder: AcquisitionFunctionBuilder | None = None,
		beta: float = 0.7,
		kappa: float = 1e-4,
	):
		"""
		:param builder: The acquisition function builder to use. :class:`TrustRegion` will attempt
			to **maximise** the corresponding acquisition function. Defaults to
			:class:`~trieste.acquisition.ExpectedImprovement` with tag `OBJECTIVE`.
		:param beta: The inverse of the trust region contraction factor.
		:param kappa: Scales the threshold for the minimal improvement required for a step to be
			considered a success.
		"""
		if builder is None:
			builder = ExpectedImprovement().using(OBJECTIVE)

		self._builder = builder
		self._beta = beta
		self._kappa = kappa

	def __repr__(self) -> str:
		""""""
		return f"TrustRegion({self._builder!r}, {self._beta!r}, {self._kappa!r})"

	def acquire(
		self,
		search_space: Box,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: State | None,
	) -> tuple[TensorType, State]:
		"""
		Acquire one new query point according the trust region algorithm. Return the new query point
		along with the final acquisition state from this step.

		If no ``state`` is specified (it is `None`), ``search_space`` is used as
		the search space for this step.

		If a ``state`` is specified, and the new optimum improves over the previous optimum
		by some threshold (that scales linearly with ``kappa``), the previous acquisition is
		considered successful.

		If the previous acquisition was successful, ``search_space`` is used as the new
		search space. If the previous step was unsuccessful, the search space is changed to the
		trust region if it was global, and vice versa.

		If the previous acquisition was over the trust region, the size of the trust region is
		modified. If the previous acquisition was successful, the size is increased by a factor
		``1 / beta``. Conversely, if it was unsuccessful, the size is reduced by the factor
		``beta``.

		**Note:** The acquisition search space will never extend beyond the boundary of the
		``search_space``. For a local search, the actual search space will be the
		intersection of the trust region and ``search_space``.

		:param search_space: The global search space for the optimization problem.
		:param datasets: The known observer query points and observations. Uses the data for key
			`OBJECTIVE` to calculate the new trust region.
		:param models: The models of the specified ``datasets``.
		:param state: The acquisition state from the previous step, if there was a previous step,
			else `None`.
		:return: A 2-tuple of the query point and the acquisition state for this step.
		:raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
		"""
		dataset = datasets[OBJECTIVE]

		global_lower = search_space.lower
		global_upper = search_space.upper

		y_min = tf.reduce_min(dataset.observations, axis=0)

		if state is None:
			eps = 0.5 * (global_upper - global_lower) / (5.0 ** (1.0 / global_lower.shape[-1]))
			is_global = True
		else:
			tr_volume = tf.reduce_prod(
				state.acquisition_space.upper - state.acquisition_space.lower
			)
			step_is_success = y_min < state.y_min - self._kappa * tr_volume

			eps = (
				state.eps
				if state.is_global
				else state.eps / self._beta
				if step_is_success
				else state.eps * self._beta
			)

			is_global = step_is_success or not state.is_global

		if is_global:
			acquisition_space = search_space
		else:
			xmin = dataset.query_points[tf.argmin(dataset.observations)[0], :]
			acquisition_space = Box(
				tf.reduce_max([global_lower, xmin - eps], axis=0),
				tf.reduce_min([global_upper, xmin + eps], axis=0),
			)

		acquisition_function = self._builder.prepare_acquisition_function(datasets, models)
		point = _optimizer.optimize(acquisition_space, acquisition_function)
		state_ = TrustRegion.State(acquisition_space, eps, y_min, is_global)

		return point, state_


class BatchAcquisitionRule(AcquisitionRule[None, SearchSpace]):
	""" Implements an acquisition rule for a batch of query points. """

	def __init__(self, num_query_points: int, builder: BatchAcquisitionFunctionBuilder):
		"""
		:param num_query_points: The number of points to acquire.
		:param builder: The acquisition function builder to use. :class:`BatchAcquisitionRule` will
			attempt to **maximise** the corresponding acquisition function.
		"""

		if not num_query_points > 0:
			raise ValueError(
				f"Number of query points must be greater than 0, got {num_query_points}"
			)

		self._num_query_points = num_query_points
		self._builder = builder

	def __repr__(self) -> str:
		""""""
		return f"BatchAcquisitionRule({self._num_query_points!r}, {self._builder!r})"

	def _vectorize_batch_acquisition(
		self, acquisition_function: BatchAcquisitionFunction
	) -> AcquisitionFunction:
		return lambda at: acquisition_function(
			tf.reshape(at, at.shape[:-1].as_list() + [self._num_query_points, -1])
		)

	def acquire(
		self,
		search_space: SearchSpace,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: None = None,
	) -> tuple[TensorType, None]:
		"""
		Return the batch of query points that optimizes the acquisition function produced by
		`builder` (see :meth:`__init__`).

		:param search_space: The global search space over which the optimization problem is defined.
		:param datasets: The known observer query points and observations.
		:param models: The models of the specified ``datasets``.
		:param state: Unused.
		:return: The batch of points to query, and `None`.
		"""
		expanded_search_space = search_space ** self._num_query_points

		batch_acquisition_function = self._builder.prepare_acquisition_function(datasets, models)
		vectorized_batch_acquisition = self._vectorize_batch_acquisition(batch_acquisition_function)

		vectorized_points = _optimizer.optimize(expanded_search_space, vectorized_batch_acquisition)
		points = tf.reshape(vectorized_points, [self._num_query_points, -1])

		return points, None


class FlowSampling(AcquisitionRule[None, SearchSpace]):
	""" Implements Flow-based sampler for choosing optimal points. """

	def __init__(self, num_search_space_samples: int, num_query_points: int, a: float, jitter: float=0.01):
		"""
		:param num_search_space_samples: The number of points at which to sample the posterior.
		:param num_query_points: The number of points to acquire.
		"""
		if not num_search_space_samples > 0:
			raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

		if not num_query_points > 0:
			raise ValueError(
				f"Number of query points must be greater than 0, got {num_query_points}"
			)

		self._num_search_space_samples = num_search_space_samples
		self._num_query_points = num_query_points
		self._a = a
		self._jitter=jitter

	def __repr__(self) -> str:
		""""""
		return f"FlowSampling({self._num_search_space_samples!r}, {self._num_query_points!r})"

	def acquire(
		self,
		search_space: SearchSpace,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: None = None,
	) -> tuple[TensorType, None]:
		"""
		Sample `num_search_space_samples` (see :meth:`__init__`) points from the
		``search_space``. Of those points, return the `num_query_points` points at which
		random samples yield the **minima** of the model posterior.

		:param search_space: The global search space over which the optimization problem
			is defined.
		:param datasets: Unused.
		:return: The `num_query_points` points to query, and `None`.
		:raise ValueError: If ``datasets`` do not contain the key `OBJECTIVE`, or it contains any
			other key.
		"""
		if datasets.keys() != {OBJECTIVE}:
			raise ValueError(
				f"dict of datasets must contain the single key {OBJECTIVE}, got keys {models.keys()}"
			)


		query_points = datasets["OBJECTIVE"].query_points
		observations = datasets["OBJECTIVE"].observations
		n = len(query_points)

		# estimate L as max pairwise L
		L = 0
		for i in range(n):
			for j in range(i+1,n):
				L = max(L,(tf.abs(observations[i]-observations[j])/(tf.norm(query_points[i]-query_points[j],axis=0))))


		d = len(search_space.lower)
		eta = tf.reduce_min(observations)
		flow_weights = self._a * (tf.abs(observations -eta) + jitter) / (L * (d-1))

		# fit flow parameters
		fantasy_query_points = fit_flow_params(query_points, observations, flow_weights)

		# perform flow
		x_flow, q = search_space.sample(self._num_search_space_samples)
		
		#select points still in domain 
		for i in range(len(search_space.lower)):
			q = q[x[:,i]>search_space.lower[i]]
			x = x[x[:,i]>search_space.lower[i]]
			q = q[x[:,i]<search_space.upper[i]]
			x = x[x[:,i]<search_space.upper[i]]

		if len(x)<self._num_query_points:
			raise ValueError(f" only {len(x)} points still in space, so cannot make batch of size B")


		if self._num_query_points==1:
			# get sampled point with highest prob
			return tf.gather(x, tf.argmax(q)), None
		else:
			return tf.gather(query_points, unique_indices), None




def fit_flow_params(query_points, observations, weights):
	fantasy_points = []

	# work out flow params
	for i in range(len(query_points)):
		# work through backwards 
		j = len(query_points)-i-1
		if i==0: # last one requires no fiddling
			fantasy_points.append(tf.expand_dims(query_points[-1],0))    
		else:
			x = tf.expand_dims(query_points[j],0)
			weight = weights[-1]
			fantasy_point = tf.expand_dims()
			x = inverse_individual_flow(x,fantasy_points[0],weight)

			for k in range(1,i):
				weight = weights[-(1+k)]
				x = inverse_individual_flow(x,fantasy_points[k],weight)
			fantasy_points.append(x)   


	fantasy_points.reverse() # built backwards
	return fantasy_points


# MAKE TF FUNCTION
def inverse_individual_flow(y,x_0,cst):
	r =  tf.norm(y - x_0,axis=1) - cst
	x = x_0+(y-x_0)/(1+cst/r)
	return x



# MAKE TF FUNCTION
def individual_flow(x,x_0, cst):
	r = tf.expand_dims(tf.sqrt(tf.reduce_sum((x-x_0)**2,1)),1)
	y = x + (x-x_0) * cst/(r)
	q = (1 + cst/(r))**(1-d)
	return y, q 


def flow(x,weights,fantasy_query_points):
	q = tf.ones((len(x),1),dtype=tf.float64)

	for i in range(len(weights)):
		fantasy_query_point = tf.expand_dims(fantasy_query_points[i], 0)
		weight = tf.expand_dims(weights[i], 0)
		x, q_temp = individual_flow(x, fantasy_query_point, weight)
		q = q * q_temp

	return x, q





class ModelFlowSampling(AcquisitionRule[None, SearchSpace]):
	""" Implements Flow-based sampler for choosing optimal points. """

	def __init__(self, num_search_space_samples: int, num_query_points: int, a: float):
		"""
		:param num_search_space_samples: The number of points at which to sample the posterior.
		:param num_query_points: The number of points to acquire.
		"""
		if not num_search_space_samples > 0:
			raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

		if not num_query_points > 0:
			raise ValueError(
				f"Number of query points must be greater than 0, got {num_query_points}"
			)

		self._num_search_space_samples = num_search_space_samples
		self._num_query_points = num_query_points
		self._a = a

	def __repr__(self) -> str:
		""""""
		return f"FlowSampling({self._num_search_space_samples!r}, {self._num_query_points!r})"

	def acquire(
		self,
		search_space: SearchSpace,
		datasets: Mapping[str, Dataset],
		models: Mapping[str, ProbabilisticModel],
		state: None = None,
	) -> tuple[TensorType, None]:
		"""
		Sample `num_search_space_samples` (see :meth:`__init__`) points from the
		``search_space``. Of those points, return the `num_query_points` points at which
		random samples yield the **minima** of the model posterior.

		:param search_space: The global search space over which the optimization problem
			is defined.
		:param datasets: Unused.
		:return: The `num_query_points` points to query, and `None`.
		:raise ValueError: If ``datasets`` do not contain the key `OBJECTIVE`, or it contains any
			other key.
		"""
		if datasets.keys() != {OBJECTIVE}:
			raise ValueError(
				f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
			)

		query_points = datasets["OBJECTIVE"].query_points
		observations = datasets["OBJECTIVE"].observations

		# estimate L as max GP grad mean


		sample_points = search_space.sample(10000)

		with tf.GradientTape() as g: # get gradients of posterior mean at samples
			g.watch(sample_points)
			mean, _ = models[OBJECTIVE].predict(sample_points)
		grads = g.gradient(mean,sample_points)
		grads_norm =  tf.norm(grads, axis=1)
		max_grads_norm = tf.reduce_max(grads_norm)

		if max_grads_norm < 1e-5: # threshold to improve numerical stability for 'flat' models
			L = 10
		else:
			L = max_grads_norm


		
		mean, var = models[OBJECTIVE].predict(query_points)
		sd = tf.sqrt(var)
		eta = tf.reduce_min(mean, axis=0)

		new_data = []
		d = len(search_space.lower)

		# define flow

		def flow(x,x_0, cst):
			r = tf.expand_dims(tf.sqrt(tf.reduce_sum((x-x_0)**2,1)),1)
			x = x + (x-x_0) * cst/(r)
			q = (1 + cst/(r))**(1-d)
			return x, q



		# work out flow params
		for i in range(len(query_points)):
			# work through backwards (remeber to swap back at end)
			j = len(query_points)-i-1
			if i==0: # last one requires no fiddling
				new_data.append(tf.expand_dims(query_points[j],0))    
			else:
				target = tf.expand_dims(query_points[j],0)
				cst = self._a * (tf.abs(mean[len(query_points)-1] - eta) + 1.5 * sd[len(query_points)-1])/ (L*(d-1))
				r_new = tf.norm(target - new_data[0],axis=1) - cst
				new_point = new_data[0]+(target-new_data[0])/(1+cst/r_new)
				
				for k in range(1,i):
					cst = self._a * (tf.abs(mean[len(query_points)-1-k] - eta) + 1.5 * sd[len(query_points)-1-k]) / (L*(d-1))
					r_new = tf.norm(new_point - new_data[k],axis=1) - cst
					new_point = new_data[k]+(new_point-new_data[k])/(1+cst/r_new)

				new_data.append(new_point)       	


		# perform flow


		x_flow = search_space.sample(self._num_search_space_samples)
		q = tf.ones((len(x_flow),1),dtype=tf.float64)


		for i in range(len(query_points)):
			cst =  self._a * (tf.abs(mean[i] - eta)+ 1.5*sd[i]) / (L*(d-1))
			x_i = new_data[len(new_data)-i-1]
			x_flow, q_temp = flow(x_flow,x_i,cst)
			q = q * q_temp

		q_kept = q
		x_kept = x_flow

		#select points still in domain 
		for i in range(len(search_space.lower)):
			q_kept = q_kept[x_kept[:,i]>search_space.lower[i]]
			x_kept = x_kept[x_kept[:,i]>search_space.lower[i]]
		for i in range(len(search_space.upper)):
			q_kept = q_kept[x_kept[:,i]<search_space.upper[i]]
			x_kept = x_kept[x_kept[:,i]<search_space.upper[i]]


		if self._num_query_points==1:
			return tf.gather(x_kept, tf.argmax(q_kept)), None
		else:
			return tf.gather(query_points, unique_indices), None