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
import logging


@beartype
def attach_console_logger(log_level: int = logging.DEBUG, remove_existing: bool = True):
    """
    Configure a console handler to redirect logging messages at the desired level to stdout.

    :param log_level:
        Set the logging level for the root optid console logger.

    :param remove_existing:
        If true then remove any existing console handlers from the logger before adding this one.
    """

    logger = logging.getLogger('optid')
    logger.setLevel(log_level)

    format_string = '%(asctime)s | %(levelname)-8s | %(name)-12s | %(funcName)s | %(message)s'

    if remove_existing:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
