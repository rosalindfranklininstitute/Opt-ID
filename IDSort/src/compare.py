'''
Takes 3 arguments:
1. The previous genome 
2. The new genome generated from the shim code
3. The path and name of the file to output eg. $IDDATA/shim1.txt
'''

import json

import numpy as np
import pandas as pd
from collections import Counter
from .genome_tools import ID_BCell

from .logging_utils import logging, getLogger, setLoggerLevel
logger = getLogger(__name__)


def process(options, args):

    if hasattr(options, 'verbose'):
        setLoggerLevel(logger, options.verbose)

    logger.debug('Starting')

    # TODO refactor arguments to use options named tuple
    # Check for unstructured arguments for inputs and output files
    if len(args) < 2:
        msg = 'Must provide at least two file paths to genomes to be compared.'
        logger.error(msg)
        raise Exception(msg)

    # Extract unstructured arguments
    g1_path, g2_path = args[:2]
    output_path      = args[2] if (len(args) > 2) else 'compare'

    try:
        logger.info('Loading ID_BCell genome 1 [%s]', g1_path)
        g1 = ID_BCell()
        g1.load(g1_path)

    except Exception as ex:
        logger.error('Failed to load ID_BCell genome 1 [%s]', g1_path, exc_info=ex)
        raise ex

    try:
        logger.info('Loading ID_BCell genome 2 [%s]', g2_path)
        g2 = ID_BCell()
        g2.load(g2_path)

    except Exception as ex:
        logger.error('Failed to load ID_BCell genome 2 [%s]', g2_path, exc_info=ex)
        raise ex

    if not options.legacy:

        if not output_path.endswith('.csv'):
            output_path = f'{output_path}.csv'

        # Attempt to load the ID json data
        try:
            logger.info('Loading ID info from json [%s]', options.id_filename)
            with open(options.id_filename, 'r') as fp:
                info = json.load(fp)

        except Exception as ex:
            logger.error('Failed to load ID info from json [%s]', options.id_filename, exc_info=ex)
            raise ex

        # Use the new human friendly style and output to CSV

        # Detect device type from it's used magnet types and adjust holder names accordingly
        magnet_types = { mag['type'] for beam in info['beams'] for mag in beam['mags'] }

        if sorted(list(magnet_types)) == sorted(['HT', 'HE', 'HH']):

            logger.info('Detected CPMU type device, using CPMU style holder names...')
            detected_device_type = 'CPMU'

            def holder_name_fn(beam, beam_index):

                # Prefix is shortened version of beam name
                prefix = 'U' if (beam['name'] == 'Top Beam') else 'L'

                slot_type = beam['mags'][beam_index]['type']

                if slot_type == 'HH':

                    # Beam index is 0-based and HH magnets start from beam_index=2
                    # Holder name should start from 001
                    index = f'{(beam_index - 1):03d}'

                elif slot_type == 'HT':

                    # HT magnets should only be at index 0 and beam_length-1
                    # Holder name should be Upstream B and Downstream B
                    if beam_index == 0:
                        index = 'UB'
                    elif beam_index == (len(beam['mags']) - 1):
                        index = 'DB'
                    else:
                        raise Exception('HT Magnet expected to be at start or end of beam!')

                elif slot_type == 'HE':

                    # HE magnets should only be at index 1 and beam_length-2
                    # Holder name should be Upstream A and Downstream A
                    if beam_index == 1:
                        index = 'UA'
                    elif beam_index == (len(beam['mags']) - 2):
                        index = 'DA'
                    else:
                        raise Exception('HE Magnet expected to be in second or penultimate beam slots!')
                else:
                    raise Exception('Unexpected magnet type!')

                # Combine beam name and holder name
                return f'{prefix}{index}'

            def orientation_name_fn(beam, beam_index, orientation):

                # Transform the unit S-axis field vector into the orientation of this holder within the ID
                major_field = np.matmul(np.array(beam['mags'][beam_index]['direction_matrix'], dtype=np.float32),
                                        np.array([0, 0, 1]))[2].astype(np.int32)

                # Lookup table for CPMU type devices maps:
                # (beam_name, major s-field direction, and flips state) to human readable direction
                lookup = {
                    ('Top Beam', -1, -1): 'Up & Front',
                    ('Top Beam',  1, -1): 'Up & Back',
                    ('Top Beam', -1,  1): 'Down & Front',
                    ('Top Beam',  1,  1): 'Down & Back',

                    ('Bottom Beam', -1,  1): 'Up & Front',
                    ('Bottom Beam',  1,  1): 'Up & Back',
                    ('Bottom Beam', -1, -1): 'Down & Front',
                    ('Bottom Beam',  1, -1): 'Down & Back',
                }

                # Select the appropriate human readable instruction
                return lookup[(beam['name'], major_field, orientation)]

        else:

            logger.info('Unable to detect type of device, using beam index as holder names...')
            detected_device_type = None

            def holder_name_fn(beam, beam_index):

                # Default holder name is just 0-based beam_index
                return f'{beam_index:03d}'

            def orientation_name_fn(beam, beam_index, orientation):

                # Default orientation name is just 1 or -1 flip state
                return orientation

        # Track the current index of each magnet type
        mag_indices = Counter()

        build_list = []
        # Process each beam in the ID json data
        for b, beam in enumerate(info['beams']):

            # Process all slots in this beam
            for beam_index, mag in enumerate(beam['mags']):

                slot_type = mag['type']
                slot_index = mag_indices[mag['type']]

                g1_mag_index, g1_mag_orientation = g1.genome.magnet_lists[slot_type][slot_index][:2]
                g2_mag_index, g2_mag_orientation = g2.genome.magnet_lists[slot_type][slot_index][:2]

                changed = ((g1_mag_index != g2_mag_index) or (g1_mag_orientation != g2_mag_orientation))
                flipped = changed and (g1_mag_index == g2_mag_index)

                build_list.append({
                    'Beam': beam['name'],
                    'Beam Index': beam_index,
                    'Holder': holder_name_fn(beam, beam_index),
                    'Type': slot_type,
                    'Original Magnet': g1_mag_index,
                    'Original Orientation': orientation_name_fn(beam, beam_index, g1_mag_orientation),
                    'Replacement Magnet': g2_mag_index,
                    'Replacement Orientation': orientation_name_fn(beam, beam_index, g2_mag_orientation),
                    'Operation': None if not changed else ('Flip' if flipped else 'Swap') })

                # Update the index to the next magnet of this type
                mag_indices[slot_type] += 1

        df_build_list = pd.DataFrame(build_list)

        if not options.full:
            # Remove rows without changes
            df_build_list = df_build_list[~df_build_list['Operation'].isnull()]

        df_build_list.to_csv(output_path, index=False)

    else:
        # Legacy output
        if not output_path.endswith('.txt'):
            output_path = f'{output_path}.txt'

        try:
            logger.info('Writing comparison to [%s]', output_path)

            with open(output_path, 'w') as output_file:
                # Write file header
                output_file.write("Type    Shim no.    Original   Orientation    Replacement    Orientation\n")

                # Process each magnet type
                for list_key in g1.genome.magnet_lists.keys():

                    # Process each magnet of type
                    for i, (g1_mag, g2_mag) in enumerate(zip(g1.genome.magnet_lists[list_key],
                                                             g2.genome.magnet_lists[list_key])):

                        # Only output magnets that do not match between files unless forced to display all
                        changed = g1_mag != g2_mag
                        if options.full or changed:

                            row_data = (list_key, i, g1_mag[0], g1_mag[1], g2_mag[0], g2_mag[1])

                            if changed:
                                logger.debug('Change found on magnet %d [%s]', i, row_data)

                            output_file.write('%2s \t%3i  \t\t%3s   \t%1i  \t\t%3s  \t\t%1i\n' % row_data)

        except Exception as ex:
            logger.error('Failed to write comparison to [%s]', output_path, exc_info=ex)
            raise ex

    logger.debug('Halting')

if __name__ == '__main__' :
    import optparse
    usage = "%prog [options] OutputFile"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-v', '--verbose', dest='verbose', help='Set the verbosity level [0-4]', default=0, type='int')
    parser.add_option("-i", "--info", dest="id_filename", help="Set the path to the id data", default=None, type="string")
    parser.add_option("-f", "--full", dest="full", help="Show output for all slots", action="store_true", default=False)
    parser.add_option("--legacy", dest="legacy", help="Use legacy output format", action="store_true", default=False)

    # TODO refactor arguments to use named values
    (options, args) = parser.parse_args()

    try:
        process(options, args)
    except Exception as ex:
        logger.critical('Fatal exception in compare::process', exc_info=ex)
