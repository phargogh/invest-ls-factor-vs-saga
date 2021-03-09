"""Stormwater Retention"""
import logging
import math
import numpy
import os
from osgeo import gdal, ogr
import pygeoprocessing
import taskgraph

from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)

# a constant nodata value to use for intermediates and outputs
NODATA = -1

ARGS_SPEC = {
    "model_name": "Stormwater Retention",
    "module": __name__,
    "userguide_html": "stormwater.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["lulc_path", "soil_group_path", "precipitation_path",
            "road_centerlines_path", "aggregate_areas_path"],
        "different_projections_ok": True
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "lulc_path": {
            "type": "raster",
            "bands": {1: {"type": "code"}},
            "required": True,
            "about": (
                "A GDAL-supported raster representing land use/land cover "
                "of the area"),
            "name": "land use/land cover"
        },
        "soil_group_path": {
            "type": "raster",
            "bands": {
                1: {
                    "type": "option_string",
                    "options": ["1", "2", "3", "4"]
                }
            },
            "required": True,
            "about": (
                "Raster map of hydrologic soil groups, where 1, 2, 3, and 4 "
                "correspond to groups A, B, C, and D respectively"),
            "name": "soil groups"
        },
        "precipitation_path": {
            "type": "raster",
            "bands": {1: {"type": "number", "units": "millimeters"}},
            "required": True,
            "about": ("Precipitation raster"),
            "name": "precipitation"
        },
        "biophysical_table": {
            "type": "csv",
            "columns": {
                "lucode": {"type": "code"},
                "is_connected": {"type": "boolean"},
                "EMC_P": {"type": "number", "units": "mg/L"},
                "EMC_N": {"type": "number", "units": "mg/L"},
                "RC_A": {"type": "ratio"},
                "RC_B": {"type": "ratio"},
                "RC_C": {"type": "ratio"},
                "RC_D": {"type": "ratio"},
                "IR_A": {"type": "ratio"},
                "IR_B": {"type": "ratio"},
                "IR_C": {"type": "ratio"},
                "IR_D": {"type": "ratio"}
            },
            "required": True,
            "about": "biophysical table",
            "name": "biophysical table"
        },
        "adjust_retention_ratios": {
            "type": "boolean",
            "required": True,
            "about": "Whether to adjust retention ratios using road centerlines",
            "name": "adjust retention ratios"
        },
        "retention_radius": {
            "type": "number",
            "units": "meters",
            "required": "adjust_retention_ratios",
            "about": "Radius around each pixel to adjust retention ratios",
            "name": "retention radius"
        },
        "dem_path": {
            "type": "raster",
            "bands": {1: {"type": "number", "units": "meters"}},
            "required": "adjust_retention_ratios",
            "about": "Digital elevation model of the area",
            "name": "digital elevation model" 
        },
        "road_centerlines_path": {
            "type": "vector",
            "fields": {},
            "required": "adjust_retention_ratios",
            "about": "Map of road centerlines",
            "name": "road centerlines"
        },
        "aggregate_areas_path": {
            "type": "vector",
            "fields": {},
            "required": False,
            "about": "Aggregation areas",
            "name": "watersheds"
        },
        "replacement_cost": {
            "type": "number",
            "units": "currency",
            "required": False,
            "about": "Replacement cost of stormwater retention devices",
            "name": "replacement cost"
        }
    }
}





def execute(args):

    # set up files and directories
    suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = args['workspace_dir']
    intermediate_dir = os.path.join(output_dir, 'intermediate')
    cache_dir = os.path.join(output_dir, 'cache_dir')
    utils.make_directories([args['workspace_dir'], intermediate_dir, cache_dir])

    FILES = {
        'lulc_aligned_path': os.path.join(intermediate_dir, f'lulc_aligned{suffix}.tif'),
        'soil_group_aligned_path': os.path.join(intermediate_dir, f'soil_group_aligned{suffix}.tif'),
        'precipitation_aligned_path': os.path.join(intermediate_dir, f'precipitation_aligned{suffix}.tif'),
        'retention_ratio_path': os.path.join(output_dir, f'retention_ratio{suffix}.tif'),
        'retention_volume_path': os.path.join(output_dir, f'retention_volume{suffix}.tif'),
        'infiltration_ratio_path': os.path.join(output_dir, f'infiltration_ratio{suffix}.tif'),
        'infiltration_volume_path': os.path.join(output_dir, f'infiltration_volume{suffix}.tif'),
        'retention_value_path': os.path.join(output_dir, f'retention_value{suffix}.tif'),
        'aggregate_data_path': os.path.join(output_dir, f'aggregate{suffix}.gpkg')
    }
    
    align_inputs = [args['lulc_path'], args['soil_group_path'], args['precipitation_path']]
    align_outputs = [
        FILES['lulc_aligned_path'],
        FILES['soil_group_aligned_path'], 
        FILES['precipitation_aligned_path']]

    pixel_size = pygeoprocessing.get_raster_info(args['lulc_path'])['pixel_size']

    task_graph = taskgraph.TaskGraph(args['workspace_dir'], int(args.get('n_workers', -1)))



    # Align all three input rasters to the same projection
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(align_inputs, align_outputs, ['near' for _ in align_inputs],
            pixel_size, 'intersection'),
        kwargs={'raster_align_index': 0},
        target_path_list=align_outputs,
        task_name='align input rasters')


    # Build a lookup dictionary mapping each LULC code to its row
    biophysical_dict = utils.build_lookup_from_csv(
        args['biophysical_table'], 'lucode')

    # Make ratio lookup dictionaries mapping each LULC code to a ratio for 
    # each soil group. Biophysical table has runoff coefficents so subtract 
    # from 1 to get retention coefficient.
    retention_ratio_dict = {
        lucode: {
            'A': 1 - row['rc_a'],
            'B': 1 - row['rc_b'],
            'C': 1 - row['rc_c'],
            'D': 1 - row['rc_d'],
        } for lucode, row in biophysical_dict.items()
    }
    infiltration_ratio_dict = {
        lucode: {
            'A': row['ir_a'],
            'B': row['ir_b'],
            'C': row['ir_c'],
            'D': row['ir_d'],
        } for lucode, row in biophysical_dict.items()
    }


    # Calculate stormwater retention ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation

    retention_ratio_task = task_graph.add_task(
        func=calculate_stormwater_ratio,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            retention_ratio_dict,
            FILES['retention_ratio_path']),
        target_path_list=[FILES['retention_ratio_path']],
        dependent_task_list=[align_task],
        task_name='calculate stormwater retention ratio'
    )

    # (Optional) adjust stormwater retention ratio using roads
    if args['adjust_retention_ratios']:

        is_connected_lookup = {lucode: row['is_connected'] for lucode, row in biophysical_dict.items()}
        connected_lulc_task = task_graph.add_task(
            func=calculate_connected_lulc_raster,
            args=(args['lulc_path'], is_connected_lookup, FILES['connected_lulc_path']),
            target_path_list=[FILES['connected_lulc_path']],
            task_name='calculate binary connected lulc raster'
        )

        # # calculate D8 flow direction from DEM
        # flow_dir_task = task_graph.add_task(
        #     func=pygeoprocessing.routing.flow_dir_d8,
        #     args=(
        #         (args['dem_path'], 1), 
        #         FILES['flow_dir_d8_path']),
        #     target_path_list=[FILES['flow_dir_d8_path']],
        #     task_name='calculate D8 flow direction'
        # )

        # connection_distance_task = task_graph.add_task(
        #     func=pygeoprocessing.routing.distance_to_channel_d8,
        #     args=(
        #         (FILES['flow_dir_d8_path'], 1),
        #         (FILES['connected_lulc_path'], 1),
        #         FILES['connection_distance_path']),
        #     target_path_list=[FILES['connection_distance_path']],
        #     task_name='calculate connection distance raster'
        # )



        # adjust_retention_ratio_task = task_graph.add_task(
        #     func=adjust_stormwater_retention_ratio,
        #     args=(
        #         FILES['retention_ratio_path'],
        #         args['road_centerlines_path'],
        #         FILES['adjusted_retention_ratio_path']),
        #     target_path_list=[FILES['adjusted_retention_ratio_path']],
        #     task_name='adjust stormwater retention ratio'
        # )
        final_retention_ratio_path = FILES['adjusted_retention_ratio_path']
    else:
        final_retention_ratio_path = FILES['retention_ratio_path']

    retention_volume_task = task_graph.add_task(
        func=calculate_stormwater_volume,
        args=(
            final_retention_ratio_path,
            FILES['precipitation_aligned_path'],
            FILES['retention_volume_path']),
        target_path_list=[FILES['retention_volume_path']],
        dependent_task_list=[align_task, retention_ratio_task],
        task_name='calculate stormwater retention volume'
    )


    # (Optional) Calculate stormwater infiltration ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation

    infiltration_ratio_task = task_graph.add_task(
        func=calculate_stormwater_ratio,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            infiltration_ratio_dict,
            FILES['infiltration_ratio_path']),
        target_path_list=[FILES['infiltration_ratio_path']],
        dependent_task_list=[align_task],
        task_name='calculate stormwater infiltration ratio'
    )

    infiltration_volume_task = task_graph.add_task(
        func=calculate_stormwater_volume,
        args=(
            FILES['infiltration_ratio_path'],
            FILES['precipitation_aligned_path'],
            FILES['infiltration_volume_path']),
        target_path_list=[FILES['infiltration_volume_path']],
        dependent_task_list=[align_task, infiltration_ratio_task],
        task_name='calculate stormwater retention volume'
    )


    # get all EMC columns from an arbitrary row in the dictionary
    # strip the first four characters off 'EMC_pollutant' to get pollutant name
    emc_columns = [key for key in next(iter(biophysical_dict.values()))
        if key.startswith('emc_')]
    pollutants = [key[4:] for key in  emc_columns]
    LOGGER.info(f'Pollutants found in biophysical table: {pollutants}')

    # Calculate avoided pollutant load for each pollutant from retention volume
    # and biophysical table EMC value
    avoided_load_paths = []
    aggregation_dependencies = [retention_volume_task, infiltration_volume_task]
    for pollutant in pollutants:
        # one output raster for each pollutant
        avoided_pollutant_load_path = os.path.join(
            output_dir, f'avoided_pollutant_load_{pollutant}{suffix}.tif')
        avoided_load_paths.append(avoided_pollutant_load_path)
        # make a dictionary mapping each LULC code to the pollutant EMC value
        lulc_emc_lookup = {
            lucode: row[f'emc_{pollutant}'] for lucode, row in biophysical_dict.items()
        }
        avoided_load_task = task_graph.add_task(
            func=calculate_avoided_pollutant_load,
            args=(
                FILES['lulc_aligned_path'],
                FILES['retention_volume_path'],
                lulc_emc_lookup,
                avoided_pollutant_load_path),
            target_path_list=[avoided_pollutant_load_path],
            dependent_task_list=[retention_volume_task],
            task_name=f'calculate avoided pollutant {pollutant} load'
        )
        aggregation_dependencies.append(avoided_load_task)


    # (Optional) Do valuation if a replacement cost is defined
    # you could theoretically have a cost of 0 which should be allowed
    print('replacement cost:', ':' + str(args['replacement_cost']) + ':', type(args['replacement_cost']), args['replacement_cost'] == '', args['replacement_cost'] in [''])
    if (args['replacement_cost'] not in [None, '']):

        valuation_task = task_graph.add_task(
            func=calculate_retention_value,
            args=(
                FILES['retention_volume_path'],
                args['replacement_cost'],
                FILES['retention_value_path']),
            target_path_list=[FILES['retention_value_path']],
            dependent_task_list=[retention_volume_task],
            task_name='calculate stormwater retention value'
        )
        aggregation_dependencies.append(valuation_task)
        valuation_path = FILES['retention_value_path']
    else:
        valuation_path = None

    # (Optional) Aggregate to watersheds if an aggregate vector is defined
    if (args['aggregate_areas_path']):
        aggregation_task = task_graph.add_task(
            func=aggregate_results,
            args=(
                args['aggregate_areas_path'],
                FILES['retention_ratio_path'],
                FILES['retention_volume_path'],
                FILES['infiltration_ratio_path'],
                FILES['infiltration_volume_path'],
                avoided_load_paths,
                valuation_path,
                FILES['aggregate_data_path']),
            target_path_list=[FILES['aggregate_data_path']],
            dependent_task_list=aggregation_dependencies,
            task_name='aggregate data over polygons'
        )

    task_graph.close()
    task_graph.join()


def calculate_stormwater_ratio(lulc_path, soil_group_path, 
        ratio_lookup, output_path):
    """Make stormwater retention or infiltration ratio map from LULC and
       soil group data.

    Args:
        lulc_path (str): path to a LULC raster whose LULC codes exist in the
            biophysical table
        soil_group_path (str): path to a soil group raster with pixel values
            1, 2, 3, and 4 corresponding to hydrologic soil groups A, B, C, and D
        ratio_lookup (dict): a lookup dictionary of ratios for each pair of 
            LULC code and soil group. Each LULC code is mapped to a dictionary
            with keys 'A', 'B', 'C', and 'D', which map to the ratio for that
            LULC code x soil group pair.
        output_path: path to write out the retention ratio raster to

    Returns:
        None
    """
    def ratio_op(lulc_array, soil_group_array):
        """Make an array of stormwater retention or infiltration ratios from 
        arrays of LULC codes and hydrologic soil groups"""

        # initialize an array of the output nodata value
        ratio_array = numpy.full(lulc_array.shape, NODATA, dtype=float)
        soil_group_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

        for lucode in ratio_lookup:
            lucode_mask = (lulc_array == lucode)

            for soil_group in [1, 2, 3, 4]:
                soil_group_mask = (soil_group_array == soil_group)
                ratio_array[lucode_mask & soil_group_mask] = ratio_lookup[lucode][soil_group_map[soil_group]]

        return ratio_array

    # Apply ratio_op to each block of the LULC and soil group rasters
    # Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(lulc_path, 1), (soil_group_path, 1)],
        ratio_op, output_path, gdal.GDT_Float32, NODATA)


def calculate_connected_lulc(lulc_path, is_connected_lookup, output_path):
    """Make a binary raster where 1=connected, 0=not
    """
    def connected_op(lulc_array):
        is_connected_array = numpy.full(lulc_array.shape, NODATA)
        for lucode in is_connected_lookup:
            lucode_mask = (lulc_array == lucode)
            is_connected_array[lucode_mask] = lulc_connected_lookup[lucode]
        return is_connected_array
    pygeoprocessing.raster_calculator(
        [(lulc_path, 1)], connected_op, output_path, gdal.GDT_Float32, NODATA)


def road_centerlines(centerlines_path):


    vector = gdal.OpenEx(centerlines_path)
    layer = aggregate_vector.GetLayer()
    for geometry in layer:
        ref = geometry.GetGeometryRef()
        assert ref.GetGeometryName() == 'LINESTRING'
        points = ref.GetPoints()  # a list of (x, y) points
        x1, y1 = points[0]
        for point in points[1:]:
            x2, y2 = point
            line_op(x1, y1, x2, y2)
            x1, y1 = x2, y2


    def line_op(x_coords, y_coords, x1, y1, x2, y2):
        """Find the minimum distance from each array point to a line segment.

        Args:
            x_coords (numpy.ndarray): a 2D array where each element is the
                x-coordinate of a point in the same coordinate system as the
                line endpoints
            y_coords (numpy.ndarray): a 2D array where each element is the
                y-coordinate of a point in the same coordinate system as the
                line endpoints
            x1 (float): the x coord of the first endpoint of the line segment
            y1 (float): the y coord of the first endpoint of the line segment
            x2 (float): the x coord of the second endpoint of the line segment
            y2 (float): the y coord of the second endpoint of the line segment

        Returns:
            numpy.ndarray with the same shape as x_coords and y_coords. The
            value of an element at [a, b] is the minimum distance from the
            point (x_coords[a, b], y_coords[a, b]) to the line segment from 
            (x1, y1) to (x2, y2). 
        """

        output_array = numpy.full(x_coords.shape, NODATA)
        distance_array = numpy.full(x_coords.shape, NODATA)

        # distance from a point to a line segment (https://math.stackexchange.com/questions/2248617/shortest-distance-between-a-point-and-a-line-segment)
        # given a point (a, b) and a line segment from (x1, y1) to (x2, y2):
        # draw a line that passes through (a, b) and is perpendicular to the 
        # line segment (negative inverse slope).
        # (a) if it intersects the line segment at a point (c, d), then the 
        #     distance is the distance from (a, b) to (c, d).
        # (b) if it doesn't intersect the line segment, then the distance is
        #     the lesser of the two distances: (a, b) -> (x1, y1) or
        #     (a, b) to (x2, y2) (this is finding the nearest endpoint)

        # solve for the line segment connecting the point (x1, y1) to (x2, y2)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # the line perpendicular to the line segment and passing thru each point
        perpendicular_slope = -1 / slope  # slope is the same for each line
        # a different intercept for each (x, y) pair
        perpendicular_intercepts = y_coords - perpendicular_slope * x_coords

        # for each (x, y) pair, the x-coordinate of the intersection of 
        # the point's perpendicular line and the original line (assuming an infinite line)
        x_intersections = (perpendicular_intercepts - intercept) / (slope - perpendicular_slope)

        # determine which points satisfy case (a) above: 
        # their intersection is within the bounds of the line segment
        within_bounds = (x_intersections >= min(x1, x2) & x_intersection <= max(x1, x2))
        # calculate their distance as the distance from (a, b) 
        # to the intersection point
        distance_array[within_bounds] = numpy.hypot(
            x_intersections - a, 
            (slope * x_intersections) + intercept - b
        )

        # for the points in case (b) above, find the minimum endpoint distance
        distance_to_endpoint_1 = numpy.hypot((x_coords - x1), (y_coords - y1))
        distance_to_endpoint_2 = numpy.hypot((x_coords - x2), (y_coords - y2))
        distance_array[~within_bounds] = numpy.min(
            distance_to_endpoint_1, distance_to_endpoint_2)

        # return a boolean array showing which points are within the radius
        return (distance_array <= radius)


def make_coordinate_arrays(raster_path):
    """Make coordinate arrays where each element is the coordinate of the
    centerpoint of a pixel in the raster coordinate system.

    Args:
        raster_path (string): raster to generate coordinates for

    Yields:
        list[tuple(numpy.ndarray, numpy.ndarray)]: For each memory block,
            a pair of 2D arrays. The first is x coords, the second is y coords.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_size_x, pixel_size_y = raster_info['pixel_size']
    x_origin = raster_info['geotransform'][0]
    y_origin = raster_info['geotransform'][3]

    for data, array in pygeoprocessing.iterblocks((raster_path, 1)):
        y_coords, x_coords = numpy.indices(array.shape)

        x_coords = (
            (x_coords * pixel_size_x) +  # convert to pixel size in meters
            (pixel_size_x / 2) +  # center the point on the pixel
            (data['xoff'] * pixel_size_x) +   # the offset of this block relative to the raster
            x_origin)  # the raster's offset relative to the coordinate system

        y_coords = (
            (y_coords * pixel_size_y) + 
            (pixel_size_y / 2) +
            (data['yoff'] * pixel_size_y) +
            y_origin)

        yield x_coords, y_coords




def adjust_stormwater_retention_ratios(retention_ratio_path, connected_path, road_centerlines_path):
    """Adjust retention ratios according to surrounding LULC and roads.

    Args:
        retention_ratio_path (str): path to raster of retention ratio values
        connected_path (str): path to a binary raster where 1 is directly
            connected impervious LULC type, 0 is not
        road_centerlines_path (str): path to line vector showing road centerlines

    Returns:
        None
    """
    
    radius = 30

    radius = 2
    # the search kernel is just large enough to contain all pixels that
    # *could* be within the radius of the center pixel
    search_kernel_shape = tuple([radius*2+1]*2)
    centerpoint = tuple([radius+1]*2)

    # arrays of the column index and row index of each pixel
    col_indices, row_indices = numpy.indices(search_kernel_shape)
    # adjust them so that (0, 0) is the center pixel
    col_index -= radius
    row_index -= radius

    # This could be expanded to flesh out the proportion of a pixel in the 
    # mask if needed, but for this convolution demo, not needed.

    # hypotenuse_i = sqrt(col_index_i**2 + row_index_i**2) for each pixel i
    hypotenuse = numpy.hypot(col_index, row_index)

    # boolean kernel where 1=pixel centerpoint is within the radius of the 
    # center pixel's centerpoint
    search_kernel = numpy.array(hypotenuse < radius, dtype=numpy.uint8)


    def adjust_op(impervious_array, retention_ratio_array):
        # total # of pixels within the search kernel that are impervious LULC
        convolved = scipy.signal.convolve(impervious_array, search_kernel)

        # boolean array where 1 = pixel is within radius of impervious LULC
        is_near_impervious = (convolved > 0)

        # array where each value is the number of valid values within the
        # search kernel. 
        # - for every kernel that doesn't extend past the edge of the original 
        #   array, this is search_kernel.size
        # - for kernels that extend past the edge, this is the number of elements
        #   that are within the original array
        n_values_array = scipy.ndimage.convolve(
            numpy.ones(retention_ratio_array.shape), 
            search_kernel, 
            mode='constant', 
            cval=0)

        # array where each pixel is averaged with its neighboring pixels within
        # the search radius. 
        averaged_ratio_array = (
            scipy.ndimage.convolve(
                retention_ratio_array, 
                search_kernel,
                mode='constant',
                cval=0
            ) / n_values_array)

        # adjustment factor:
        # - 0 if any of the nearby pixels are impervious/connected;
        # - average of nearby pixels, otherwise
        adjustment_factor_array = averaged_ratio_array * ~is_near_impervious

        # equation 2-4
        adjusted_ratio_array = (retention_ratio_array + 
            (1 - retention_ratio_array) * adjustment_factor_array)

        return adjusted_ratio_array



        

        



def calculate_stormwater_volume(ratio_path, precipitation_path, output_path):
    """Make stormwater retention or infiltration volume map from ratio and 
       precipitation.

    Args:
        ratio_path (str): path to a raster of stormwater ratios
        precipitation_path (str): path to a raster of precipitation amounts
        output_path (str): path to write out the volume results (raster)

    Returns:
        None
    """
    ratio_raster_info = pygeoprocessing.get_raster_info(ratio_path)
    ratio_nodata = ratio_raster_info['nodata'][0]
    pixel_area = abs(ratio_raster_info['pixel_size'][0] * 
        ratio_raster_info['pixel_size'][1])
    precipitation_nodata = pygeoprocessing.get_raster_info(
        precipitation_path)['nodata'][0]

    def volume_op(ratio_array, precipitation_array):
        """Calculate array of volumes (retention or infiltration) from arrays 
        of precipitation values and stormwater ratios"""

        volume_array = numpy.full(ratio_array.shape, NODATA, dtype=float)
        nodata_mask = (
            (ratio_array != ratio_nodata) & 
            (precipitation_array != precipitation_nodata))

        # precipitation (mm/yr) * pixel area (m^2) * 
        # 0.001 (m/mm) * ratio = volume (m^3/yr)
        volume_array[nodata_mask] = (
            precipitation_array[nodata_mask] *
            ratio_array[nodata_mask] *
            pixel_area * 0.001)
        return volume_array

    # Apply volume_op to each block in the ratio and precipitation rasters
    # Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(ratio_path, 1), (precipitation_path, 1)],
        volume_op, output_path, gdal.GDT_Float32, NODATA)



def calculate_avoided_pollutant_load(lulc_path, retention_volume_path, 
        lulc_emc_lookup, output_path):
    """Make avoided pollutant load map from retention volumes and LULC event 
       mean concentration data.

    Args:
        lulc_path (str): path to a LULC raster whose LULC codes exist in the
            EMC lookup dictionary
        retention_volume_path: (str) path to a raster of stormwater retention
            volumes in m^3
        lulc_emc_lookup (dict): a lookup dictionary where keys are LULC codes 
            and values are event mean concentration (EMC) values in mg/L for 
            the pollutant in that LULC area.
        output_path (str): path to write out the results (raster)

    Returns:
        None
    """
    lulc_nodata = pygeoprocessing.get_raster_info(lulc_path)['nodata'][0]


    def avoided_pollutant_load_op(lulc_array, retention_volume_array):
        """Calculate array of avoided pollutant load values from arrays of 
        LULC codes and stormwater retention volumes."""

        load_array = numpy.full(lulc_array.shape, NODATA, dtype=float)

        nodata_mask = (
            (lulc_array != lulc_nodata) &
            (retention_volume_array != NODATA))

        for lucode in lulc_emc_lookup:
            lucode_mask = (lulc_array == lucode)
            # EMC for pollutant (mg/L) * 1000 (L/m^3) * 0.000001 (kg/mg) * 
            # retention (m^3/yr) = pollutant load (kg/yr)
            load_array[lucode_mask & nodata_mask] = (
                lulc_emc_lookup[lucode] * 0.001 * 
                retention_volume_array[lucode_mask & nodata_mask])
        return load_array

    # Apply avoided_pollutant_load_op to each block of the LULC and retention 
    # volume rasters. Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(lulc_path, 1), (retention_volume_path, 1)],
        avoided_pollutant_load_op, output_path, gdal.GDT_Float32, NODATA)


def calculate_retention_value(retention_volume_path, replacement_cost, output_path):
    """Calculate retention value from retention volume and replacement cost.
    Args:
        retention_volume_path (str): path to retention volume raster (m^3/pixel)
        replacement_cost (float): value in currency units/m^3
        output_path (str): path to write out valuation results raster

    Returns:
        None
    """
    def retention_value_op(retention_volume_array):
        """Multiply array of retention volumes by the retention replacement 
        cost to get an array of retention values."""
        value_array = numpy.full(retention_volume_array.shape, NODATA, dtype=float)
        nodata_mask = (retention_volume_array != NODATA)

        # retention (m^3/yr) * replacement cost ($/m^3) = retention value ($/yr)
        value_array[nodata_mask] = (
            retention_volume_array[nodata_mask] * replacement_cost)
        return value_array

    # Apply retention_value_op to each block of the retention volume rasters
    # Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(retention_volume_path, 1)],
        retention_value_op, output_path, gdal.GDT_Float32, NODATA)



def aggregate_results(aoi_path, r_ratio_path, r_volume_path, 
        i_ratio_path, i_volume_path, avoided_pollutant_loads, 
        retention_value, output_path):
    """Aggregate outputs into regions of interest.

    Args:
        aoi_path (str): path to vector of polygon(s) to aggregate over
        retention_ratio (str): path to stormwater retention ratio raster
        retention_volume (str): path to stormwater retention volume raster
        infiltration_ratio (str): path to stormwater infiltration ratio raster
        infiltration_volume (str): path to stormwater infiltration volume raster
        avoided_pollutant_loads (list[str]): list of paths to avoided pollutant
            load rasters
        retention_value (str): path to retention value raster
        output_path (str): path to write out aggregated vector data

    Returns:
        None
    """

    if os.path.exists(output_path):
        LOGGER.warning(
            '%s exists, deleting and writing new output',
            output_path)
        os.remove(output_path)

    original_aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)

    # copy AOI vector to the output path and convert to GPKG if needed
    result = gdal.VectorTranslate(output_path, aoi_path)
    
    aggregate_vector = gdal.OpenEx(output_path, 1)
    aggregate_layer = aggregate_vector.GetLayer()

    aggregations = [
        (r_ratio_path, 'RR_mean', 'mean'),     # average retention ratio
        (r_volume_path, 'RV_sum', 'sum'),      # total retention volume
        (i_ratio_path, 'IR_mean', 'mean'),     # average infiltration ratio
        (i_volume_path, 'IV_sum', 'sum'),      # total infiltration volume
    ]
    if (retention_value):                      # total retention value
        aggregations.append((retention_value, 'val_sum', 'sum'))
    for avoided_load_path in avoided_pollutant_loads:
        pollutant = avoided_load_path.split('_')[-1]
        field = f'avoided_{pollutant}'
        aggregations.append((avoided_load_path, field, 'sum'))


    for raster_path, field_id, op in aggregations:
        # aggregate the raster by the vector region(s)
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), output_path)

        # set up the field to hold the aggregate data
        aggregate_field = ogr.FieldDefn(field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)
        aggregate_layer.ResetReading()

        # save the aggregate data to the field for each feature
        for polygon in aggregate_layer:
            feature_id = polygon.GetFID()
            if op == 'mean':
                pixel_count = aggregate_stats[feature_id]['count']
                if pixel_count != 0:
                    value = (aggregate_stats[feature_id]['sum'] / pixel_count)
                else:
                    LOGGER.warning(
                        "no coverage for polygon %s", ', '.join(
                            [str(polygon.GetField(_)) for _ in range(
                                polygon.GetFieldCount())]))
                    value = 0.0
            elif op == 'sum':
                value = aggregate_stats[feature_id]['sum']
            polygon.SetField(field_id, float(value))
            aggregate_layer.SetFeature(polygon)

    # save the aggregate vector layer and clean up references
    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None

    
@validation.invest_validator
def validate(args):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    return validation.validate(args, ARGS_SPEC['args'],
                               ARGS_SPEC['args_with_spatial_overlap'])



















