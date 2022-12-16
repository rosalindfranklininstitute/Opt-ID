# Copyright 2017 Diamond Light Source
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


# External Imports
from beartype import beartype
import beartype.typing as typ
from functools import cached_property


@beartype
def invalidates_cached_properties(
        func: typ.Callable,
        before: bool = True,
        after: bool = True) -> typ.Callable:

    @beartype
    def wrap(self: Memoized, *args, **kargs) -> typ.Any:

        if before:
            self.invalidate_cached_properties()

        result = func(self, *args, **kargs)

        if after:
            self.invalidate_cached_properties()

        return result

    return wrap


def find_all_cached_properties(cls):

    for base in cls.__bases__:
        yield from find_all_cached_properties(base)

    for key, value in cls.__dict__.items():
        if isinstance(value, cached_property):
            yield key, value


class Memoized:

    def invalidate_cached_properties(self):

        for key, value in find_all_cached_properties(type(self)):
            if key in self.__dict__:
                del self.__dict__[key]
