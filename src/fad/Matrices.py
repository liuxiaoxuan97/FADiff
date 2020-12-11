#!/usr/bin/env python3

from FADiff import FADiff
import numpy as np


class Vect:
    """
   A class for automatic differentiation of vector variables (NumPy arrays)
   """

    def __init__(self, val, der=None, parents=None, name=None, new_input=False):
        """
                Inputs
                ------
                    val : NumPy array (column or row vector)
                        value of the vector variable
                    der : NumPy array (column or row vector) or a dictionary
                        derivative of the vector variable
                    parents : list of Scal objects
                        the parent/grandparent vars of the variable
                    name : str
                        the name of the variable
                    new_input : boolean
                        if variable is an input variable
                """

        # preprocess inputs
        if new_input:
            if len(val.shape) > 1 and min(val.shape) > 1:  # check for correct shape
                raise TypeError('Val and Der must be row or column vectors (NumPy arrays) of same shape.')
            elif len(der.shape) > 1 and min(der.shape) > 1:  # check for correct shape
                raise TypeError('Val and Der must be row or column vectors (NumPy arrays) of same shape.')
            else:
                value = val.reshape(1, -1)
                deriv = der.reshape(1, -1)
                zero = np.zeros(value.shape)
        else:
            value = val
            deriv = der

        self._val = value
        if new_input:  # Creating input var?
            self._der = {}  # Add gradient dict for new var
            for var in FADiff._fadvect_inputs:  # Update gradient dicts for all vars
                self._der[var] = zero  # Partial der of others as 0 in self
                var._der[self] = zero  # Self's partial der as 0 in others
            self._der[self] = deriv  # Self's partial der in self
            FADiff._fadvect_inputs.append(self)  # Add self to global vars list
        else:
            self._der = deriv
        self._name = name
        if parents is None:
            parents = []
        self._parents = parents

    ### Basic Operations ###

    def __add__(self, other):
        """
        Adds self with other (self + other)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = self._val + other._val
            der = {}
            for var, part_der in self._der.items():
                der[var] = part_der + other._der.get(var)
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = self._val + other
            der = self._der
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __radd__(self, other):
        """
        Adds other with self (other + self)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtracts other from self (self - other)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = self._val - other._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = part_der - other._der.get(var)
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = self._val - other
            der = self._der
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __rsub__(self, other):
        """
        Subtracts self from other (other - self)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = other._val - self._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = other._der.get(var) - part_der
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = other - self._val
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = other - part_der
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __mul__(self, other):
        """
        Multiplies self with other (self * other)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = self._val * other._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = part_der * other._val + self._val * other._der.get(var)
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = self._val * other
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = part_der * other
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __rmul__(self, other):
        """
        Multiplies other with self (other * self)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divides self by other (self / other)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = self._val / other._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = (part_der * other._val - self._val * other._der.get(var)) / (other._val * other._val)
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = self._val / other
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = part_der / other
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __rtruediv__(self, other):
        """
        Divides other by self (other / self)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = other._val / self._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = (self._val * other._der.get(var) - part_der * other._val) / (self._val * self._val)
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = other / self._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = (- other / (self._val * self._val)) * part_der
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __pow__(self, other):
        """
        Raises self the other power (self ** other)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = self._val ** other._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = other._val * (self._val ** (other._val - 1.)) * part_der
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = self._val ** other
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = other * (self._val ** (other - 1.)) * part_der
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __rpow__(self, other):
        """
        Raises other the self power (other ** self)

        Inputs: self (Scal object), other (either Scal object or constant)
        Returns: new Scal object
        """
        try:  # if other is a Scal
            val = other._val ** self._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = (other._val ** self._val) * np.log(other._val) * part_der
            parents = self._set_parents(self, other)
        except AttributeError:  # if other is a constant
            val = other ** self._val
            der = {}
            for var, part_der in self._der.items():  # loop through partial derivatives
                der[var] = (other ** self._val) * np.log(other) * part_der
            parents = self._set_parents(self)
        return Vect(val, der, parents)

    def __neg__(self):
        """
        Negates self (- self)

        Inputs: self (Scal object)
        Returns: new Scal object
        """
        val = -self._val
        der = {}
        for var, part_der in self._der.items():
            der[var] = - part_der
        parents = self._set_parents(self)
        return Vect(val, der, parents)

    ### Comparison Operators ###

    def __eq__(self, other):
        if isinstance(other, Vect):
            return self.__key() == other.__key()
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Vect):
            return self.__key() != other.__key()
        return NotImplemented

    def __key(self):
        return id(self)

    def __hash__(self):
        return hash(self.__key())

    @property
    def val(self):
        """
        Inputs: self (Scal object)
        Returns: NumPy array of values
        """
        return np.array(self._val)

    @property
    def der(self):
        """
        Returns partial derivatives wrt all root input vars used

        Inputs: self (Scal object)
        Returns: NumPy array of derivative
        """
        parents = []
        for var, part_der in self._der.items():
            if var in self._parents:
                parents.append(part_der)
        if parents:  # For output vars
            return np.array(parents)
        elif self in FADiff._fadvect_inputs:  # For input vars (no parents)
            return np.array(self._der[self])

    @staticmethod
    def _set_parents(var1, var2=None):
        '''Sets parent/grandparent vars (including root input vars used)'''
        parents = []
        parents.append(var1)
        for parent in var1._parents:
            parents.append(parent)
        if var2:
            parents.append(var2)
            for parent in var2._parents:
                parents.append(parent)
        parents = list(set(parents))
        return parents
