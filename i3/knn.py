#!/usr/bin/env python

"""
This module provides classes for training and using
k-nearest-neighbors on numeric vectors and structured objects.
"""

import collections

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    print("Could not import sklearn. SciKitKNNGetter not available.")

from invert.distance import structured_distance
from invert.structure import get_nested_structure_and_fields
from invert.utils import logger

LOG = logger(__name__)

__all__ = ['StructuredKNNGetter',
           'SciKitKNNGetter',
           'BruteKNNGetter']


class KNNGetter(object):
    """
    An interface for objects that get k nearest neighbors.
    """

    def train(self, pairs):
        """
        Adds (x, y) pairs to the neighborhood.
        """
        raise NotImplementedError('train')

    def knns(self, x, k):
        """
        Gets the k nearest pairs to k.
        """
        raise NotImplementedError('knns')


class BruteKNNGetter(KNNGetter):
    """
    Gets the k nearest neighbors by computing distance to all xs.

    Attributes:
        objects: A list of (x, y) pairs.
        distance_metric: A function from a pair of values to the
            distance between them.
    """

    def __init__(self, distance_metric):
        self.objects = []
        self.distance_metric = distance_metric

    def train(self, pairs):
        self.objects.extend(pairs)

    def knns(self, x, k):
        objs_with_distance = [(o, self.distance_metric(x, o[0]))
                              for o in self.objects]
        objs_with_distance.sort(key=lambda od: od[1])
        knns = objs_with_distance[:k]
        if knns:
            # Keep all neighbors (perhaps more than k) that are tied
            # for the kth nearest neigbor.
            for i in range(k, len(objs_with_distance)):
                if objs_with_distance[i][1] == knns[-1][1]:
                    knns.append(objs_with_distance[i])
            return zip(*knns)
        else:
            return [], []


class SciKitKNNGetter(KNNGetter):
    """
    Computes k nearest neighbors using scikit.

    Only works for numeric vector x values.

    Attributes:
        neighborhoods: Maps nested structure to NearestNeighbors instance for
            that structure.
    """

    def __init__(self):
        self.pairs = []
        self.nn = None

    def train(self, pairs):
        self.pairs.extend(pairs)
        self.nn = None

    def knns(self, x, k):
        if not self.pairs:
            return [], []
        if not self.nn:
            xs = [xi for (xi, _) in self.pairs if xi]
            if not xs:
                return [], []
            self.nn = NearestNeighbors()
            self.nn.fit(xs)
        distance_array, index_array = self.nn.kneighbors(
                x, min(k, len(self.pairs)), return_distance=True)
        distances = list(distance_array[0])
        indices = list(index_array[0])
        elements = [self.pairs[i] for i in indices]
        return elements, distances


class StructuredKNNGetter(KNNGetter):
    """
    KNN getter for arbitrary structured objects.

    Contains one SciKitKNNGetter for every distinct nested structure.
    """

    def __init__(self):
        self.vector_knns = collections.defaultdict(lambda: SciKitKNNGetter())
        self.backup = BruteKNNGetter(structured_distance)

    def train(self, pairs):
        for (x, y) in pairs:
            x_struct, x_fields = get_nested_structure_and_fields(x)
            self.vector_knns[x_struct].train([(x_fields, y)])
        self.backup.train(pairs)

    def knns(self, x, k):
        x_struct, x_fields = get_nested_structure_and_fields(x)
        knns, distances = self.vector_knns[x_struct].knns(x_fields, k)
        if len(knns) >= k:
            return knns, distances
        return self.backup.knns(x, k)
